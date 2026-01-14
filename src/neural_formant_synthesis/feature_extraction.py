import torch
import diffsptk
import numpy as np
from neural_formant_synthesis.functions import frame_energy, spectral_centroid, tilt_levinson, pitch_extraction
from neural_formant_synthesis.glotnet.sigproc.emphasis import Emphasis

class feature_extractor(torch.nn.Module):
    """
    Class to extract the features for neural formant synthesis.
    Params:
        sr: sample rate
        window_samples: window size in samples
        step_samples: step size in samples
        formant_ceiling: formant ceiling in Hz
        max_formants: maximum number of formants to estimate
        allpole_order: allpole order for the LPC analysis
    """
    def __init__(self, sr, window_samples, step_samples, formant_ceiling, max_formants, allpole_order = 10, r_coeff_order = 30):
        super().__init__()
        self.sr = sr
        self.window_samples = window_samples
        self.step_samples = step_samples
        self.formant_ceiling = formant_ceiling
        self.max_formants = max_formants
        self.allpole_order = allpole_order
        self.r_coeff_order = r_coeff_order

        self.framer = diffsptk.Frame(
            frame_length = self.window_samples,
            frame_period = self.step_samples
        )
        self.windower = diffsptk.Window(
            in_length = self.window_samples
        )
        self.pre_emphasis = Emphasis(alpha=0.97)
        
    def forward(self, x):
        # Signal windowing
        x_frame = self.framer(x)
        x_frame_emphasis = self.framer(self.pre_emphasis(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0))
        x_window = self.windower(x_frame)
        x_window_emphasis = self.windower(x_frame_emphasis)

        # STFT

        x_spec = torch.fft.rfft(x_window_emphasis, dim = -1)
        x_us_acorr = torch.fft.irfft(x_spec * torch.conj(x_spec), dim = -1)
        
        # Calculate other features
        energy = 10 * torch.log10(frame_energy(x_frame))
        centroid = spectral_centroid(x_window, self.sr)
        tilt = tilt_levinson(x_us_acorr)

        # Extract pitch from audio signal
        formants, pitch, voicing = pitch_extraction(x = x, sr = self.sr, window_samples = self.window_samples, step_samples = self.step_samples, fmin = 50, fmax = 500)
        formants = formants[:len(energy), ...]
        pitch = pitch[..., :len(energy)]
        voicing = voicing[:len(energy)]
        
        tilt = tilt.numpy()
        try:
            tilt[np.isnan(tilt)] = np.interp(
                    np.where(np.isnan(tilt))[0],
                    np.where(~np.isnan(tilt))[0],
                    tilt[~np.isnan(tilt)])
        except:
            pass
        tilt = torch.from_numpy(tilt)
        
        return formants, energy, centroid, tilt, pitch, voicing

class MedianPool1d(torch.nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel
         stride: pool stride
         padding: pool padding
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool1d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.same = same

    def _padding(self, x):
        if self.same:
            iw = x.size()[-1]
            if iw % self.stride == 0:
                pw = max(self.k - self.stride, 0)
            else:
                pw = max(self.k - (iw % self.stride), 0)
            pl = pw // 2
            pr = pw - pl

            padding = (pl, pr)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        x = torch.nn.functional.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k, self.stride)
        x = x.contiguous().view(x.size()[:-1] + (-1,)).median(dim=-1)[0]
        return x
    
class Normaliser(torch.nn.Module):
    """
    Normalisation of features to a set of hard limits for training with online feature extraction.
    Params:
        sample_rate: sample rate
        pitch_lims: tuple with pitch limits in log(Hz) (lower, upper)
        formant_lims: tuple with formant limits in Hz (f1_lower, f1_upper, f2_lower, f2_upper, f3_lower, f3_upper, f4_lower, f4_upper)
        tilt_lims: tuple with spectral tilt limits (lower, upper)
        centroid_lims: tuple with spectral centroid limits in Hz (lower, upper)
        energy_lims: tuple with energy limits in dB (lower, upper)
    """
    def __init__(self, pitch_lims = None, formant_lims = None, tilt_lims = None, centroid_lims = None, energy_lims = None):
        super().__init__()
        if pitch_lims is not None:
            self.pitch_lower = pitch_lims[0]
            self.pitch_upper = pitch_lims[1]
        else:
            self.pitch_lower = 5.64385618977
            self.pitch_upper = 8.96578428466

        if formant_lims is not None:
            self.f1_lower = formant_lims[0]
            self.f1_upper = formant_lims[1]
            self.f2_lower = formant_lims[2]
            self.f2_upper = formant_lims[3]
            self.f3_lower = formant_lims[4]
            self.f3_upper = formant_lims[5]
        else:
            self.f1_lower = 8.07681559705
            self.f1_upper = 9.81378119122
            self.f2_lower = 9.81378119122
            self.f2_upper = 11.4512111118
            self.f3_lower = 11.1674181458
            self.f3_upper = 11.731319031
        
        if tilt_lims is not None:
            self.tilt_lower = tilt_lims[0]
            self.tilt_upper = tilt_lims[1]
        else:
            self.tilt_lower = -1
            self.tilt_upper = 1

        if centroid_lims is not None:
            self.centroid_lower = centroid_lims[0]
            self.centroid_upper = centroid_lims[1]
        else:
            self.centroid_lower = 0
            self.centroid_upper = 22050

        if energy_lims is not None:
            self.energy_lower = energy_lims[0]
            self.energy_upper = energy_lims[1]
        else:
            self.energy_lower = -40
            self.energy_upper = 0

    def forward(self, pitch, formants, tilt, centroid, energy):
        pitch = self._scale(torch.log2(pitch), self.f3_upper, self.pitch_lower)
        formants[...,0] = self._scale(torch.log2(formants[...,0]),self.f3_upper, self.pitch_lower)
        formants[...,1] = self._scale(torch.log2(formants[...,1]),self.f3_upper, self.pitch_lower)
        formants[...,2] = self._scale(torch.log2(formants[...,2]),self.f3_upper, self.pitch_lower)
        tilt = self._scale(tilt, self.tilt_upper, self.tilt_lower)
        centroid = self._scale(centroid, self.centroid_upper, self.centroid_lower)
        energy = self._scale(torch.clamp(energy, self.energy_lower), self.energy_upper, self.energy_lower)
        return pitch, formants, tilt, centroid, energy

    def _scale(self,feature, upper, lower):
        max_denorm = upper - lower
        feature = (2 * (feature - lower) / max_denorm) - 1
        return feature