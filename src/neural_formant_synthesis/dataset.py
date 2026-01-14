import random
import torch
import torchaudio as ta
from torch.utils.data import Dataset
from .feature_extraction import Normaliser
import io

from neural_formant_synthesis.third_party.hifi_gan.utils import mel_spectrogram

from neural_formant_synthesis.feature_extraction import feature_extractor, MedianPool1d
from neural_formant_synthesis.glotnet.sigproc.emphasis import Emphasis

import json
import os.path as osp
    
class FeatureDataset_List(Dataset):
    def __init__(self, 
                 config,
                 sampling_rate: int,
                 frame_size: int,
                 hop_size:int,
                 feature_ext: str = '.pt',
                 audio_ext: str = '.wav',
                 segment_length: int = None,
                 normalise: bool = True,
                 shuffle: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        
        self.config = config
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.feature_ext = feature_ext
        self.audio_ext = audio_ext
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.shuffle = shuffle

        self.segment_length = segment_length

        self.get_file_list()
        
        self.feat_extractor = feature_extractor(sr = self.sampling_rate, window_samples = self.frame_size, step_samples = self.hop_size, formant_ceiling = 10000, max_formants = 3)
        self.median_filter = MedianPool1d(kernel_size = 7, stride = 1, padding = 0, same = True)
        self.pre_emphasis = Emphasis(alpha=0.97)
        self.normaliser = Normaliser()
        
        self.length = len(self.file_list)

    def get_file_list(self):
        self.file_list = []
        
        self.dataset_json = json.load(open("dataset/train.json", 'r'))
        self.dataset_list = self.dataset_json['datasets']

        self.filelist_path = self.dataset_json['filelist_path']
        
        for dataset in self.dataset_list:
            filelist_json = osp.join(self.filelist_path, dataset['name'] + '.json')
            wav_data_list = json.load(open(filelist_json, 'r'))
            for wav_data in wav_data_list:
                if 'data_path' in wav_data:
                    wav = wav_data['data_path']
                elif 'path' in wav_data:
                    wav = wav_data['path']
                elif 'id' in wav_data:
                    wav = wav_data['id'] + '.wav' 
                if dataset['path'] != "":
                    wav_path = osp.join(dataset['path'], wav)
                else:
                    wav_path = wav
                    
                self.file_list.append(wav_path)
        
        if self.shuffle:
            random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        while True:
            try:
                batch = self.get_batch(index)
                break
            except Exception as e:
                index = random.randint(0, self.length - 1)
        return batch
    
    def get_batch(self, index):
        audio_path = self.file_list[index]
        audio, sample_rate = ta.load(audio_path)
        audio = ta.functional.resample(audio, sample_rate, self.sampling_rate)
        audio_total_len = audio.shape[-1] + (self.hop_size - audio.shape[-1] % self.hop_size)
        audio = torch.unsqueeze(torch.nn.functional.pad(audio.squeeze(0), (0,int(audio_total_len - audio.size(1))),'constant'),0)
        audio_total_len = audio_total_len // self.hop_size
        
        if self.segment_length <= audio_total_len:
            max_segment_start = audio_total_len - self.segment_length
            segment_start = random.randint(0, max_segment_start)
            audio_start = int(segment_start * self.hop_size)
            audio_segment_len = int(self.segment_length * self.hop_size)
            audio = audio[:,audio_start:audio_start + audio_segment_len]
        else:
            diff = self.segment_length - audio_total_len
            audio_segment_diff = int(self.hop_size * diff)
            audio = torch.unsqueeze(torch.nn.functional.pad(audio.squeeze(0), (0,audio_segment_diff),'constant'),0)

        x = audio[0:1].type(torch.DoubleTensor)
        x = x.squeeze(0)
        formants, energy, centroid, tilt, pitch, voicing_flag = self.feat_extractor(x)
        formants = self.median_filter(formants.T.unsqueeze(1)).squeeze(1).T
        
        pitch = pitch.squeeze(0)
        voicing_flag = voicing_flag.squeeze(0)
        linear_pitch = pitch
        pitch, formants, tilt, centroid, energy = self.normaliser(pitch, formants, tilt, centroid, energy)
        
        if torch.any(torch.isnan(pitch)):
            raise ValueError("Output pitch features are NaN.")
        if torch.any(torch.isnan(formants)):
            raise ValueError("Output formants features are NaN.")
        if torch.any(torch.isnan(tilt)):
            raise ValueError("Output tilt features are NaN.")
        if torch.any(torch.isnan(centroid)):
            raise ValueError("Output centroid features are NaN.")
        if torch.any(torch.isnan(energy)):
            raise ValueError("Output energy features are NaN.")
        if torch.any(torch.isnan(voicing_flag)):
            raise ValueError("Output voicing_flag features are NaN.")
        
        if not torch.all(torch.isfinite(pitch)):
            raise ValueError("Output pitch features are INF.")
        if not torch.all(torch.isfinite(formants)):
            raise ValueError("Output formants features are INF.")
        if not torch.all(torch.isfinite(tilt)):
            raise ValueError("Output tilt features are INF.")
        if not torch.all(torch.isfinite(centroid)):
            raise ValueError("Output centroid features are INF.")
        if not torch.all(torch.isfinite(energy)):
            raise ValueError("Output energy features are INF.")
        if not torch.all(torch.isfinite(voicing_flag)):
            raise ValueError("Output voicing_flag features are INF.")
        
        x = torch.transpose(torch.cat((pitch.unsqueeze(1), formants, tilt.unsqueeze(1), centroid.unsqueeze(1), energy.unsqueeze(1), voicing_flag.unsqueeze(1)),dim = -1), 0, 1)

        if torch.any(torch.isnan(audio)):
            raise ValueError("Output audio features are NaN.")
        
        if x.shape[-1] != self.segment_length:
            raise ValueError("Wrong x length.")
        if audio.shape[-1] != int(self.segment_length * self.hop_size):
            raise ValueError("Wrong audio length.")
        if linear_pitch.shape[-1] != self.segment_length:
            raise ValueError("Wrong mel length.")

        return x.type(torch.FloatTensor).to(self.device), audio.squeeze(0).type(torch.FloatTensor).to(self.device), linear_pitch.squeeze().type(torch.FloatTensor).to(self.device)
    
class FeatureDataset_Baseline(Dataset):
    def __init__(self, 
                 config,
                 sampling_rate: int,
                 frame_size: int,
                 hop_size:int,
                 feature_ext: str = '.pt',
                 audio_ext: str = '.wav',
                 segment_length: int = None,
                 normalise: bool = True,
                 shuffle: bool = False,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):
        
        self.config = config
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.feature_ext = feature_ext
        self.audio_ext = audio_ext
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.shuffle = shuffle

        self.segment_length = segment_length

        self.get_file_list()
        
        self.feat_extractor = feature_extractor(sr = self.sampling_rate, window_samples = self.frame_size, step_samples = self.hop_size, formant_ceiling = 10000, max_formants = 3)
        self.median_filter = MedianPool1d(kernel_size = 7, stride = 1, padding = 0, same = True)
        self.pre_emphasis = Emphasis(alpha=0.97)
        self.normaliser = Normaliser()
        
        self.length = len(self.file_list)

    def get_file_list(self):
        self.file_list = []
        
        self.dataset_json = json.load(open("dataset/train.json", 'r'))
        self.dataset_list = self.dataset_json['datasets']

        self.filelist_path = self.dataset_json['filelist_path']
        
        for dataset in self.dataset_list:
            filelist_json = osp.join(self.filelist_path, dataset['name'] + '.json')
            wav_data_list = json.load(open(filelist_json, 'r'))
            for wav_data in wav_data_list:
                if 'data_path' in wav_data:
                    wav = wav_data['data_path']
                elif 'path' in wav_data:
                    wav = wav_data['path']
                elif 'id' in wav_data:
                    wav = wav_data['id'] + '.wav' 
                if dataset['path'] != "":
                    wav_path = osp.join(dataset['path'], wav)
                else:
                    wav_path = wav
                    
                self.file_list.append(wav_path)
        
        if self.shuffle:
            random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        while True:
            try:
                batch = self.get_batch(index)
                break
            except Exception as e:
                index = random.randint(0, self.length - 1)
        return batch
    
    def get_batch(self, index):
        audio_path = self.file_list[index]
        audio, sample_rate = ta.load(audio_path)
        audio = ta.functional.resample(audio, sample_rate, self.sampling_rate)
        audio_total_len = audio.shape[-1] + (self.hop_size - audio.shape[-1] % self.hop_size)
        audio = torch.unsqueeze(torch.nn.functional.pad(audio.squeeze(0), (0,int(audio_total_len - audio.size(1))),'constant'),0)
        audio_total_len = audio_total_len // self.hop_size
        
        if self.segment_length <= audio_total_len:
            max_segment_start = audio_total_len - self.segment_length
            segment_start = random.randint(0, max_segment_start)
            audio_start = int(segment_start * self.hop_size)
            audio_segment_len = int(self.segment_length * self.hop_size)
            audio = audio[:,audio_start:audio_start + audio_segment_len]
        else:
            diff = self.segment_length - audio_total_len
            audio_segment_diff = int(self.hop_size * diff)
            audio = torch.unsqueeze(torch.nn.functional.pad(audio.squeeze(0), (0,audio_segment_diff),'constant'),0)

        x = audio[0:1].type(torch.DoubleTensor)
        x = x.squeeze(0)
        formants, energy, centroid, tilt, pitch, voicing_flag = self.feat_extractor(x)
        formants = self.median_filter(formants.T.unsqueeze(1)).squeeze(1).T
        
        pitch = pitch.squeeze(0)
        voicing_flag = voicing_flag.squeeze(0)
        pitch, formants, tilt, centroid, energy = self.normaliser(pitch, formants, tilt, centroid, energy)
        
        if torch.any(torch.isnan(pitch)):
            raise ValueError("Output pitch features are NaN.")
        if torch.any(torch.isnan(formants)):
            raise ValueError("Output formants features are NaN.")
        if torch.any(torch.isnan(tilt)):
            raise ValueError("Output tilt features are NaN.")
        if torch.any(torch.isnan(centroid)):
            raise ValueError("Output centroid features are NaN.")
        if torch.any(torch.isnan(energy)):
            raise ValueError("Output energy features are NaN.")
        if torch.any(torch.isnan(voicing_flag)):
            raise ValueError("Output voicing_flag features are NaN.")
        
        if not torch.all(torch.isfinite(pitch)):
            raise ValueError("Output pitch features are INF.")
        if not torch.all(torch.isfinite(formants)):
            raise ValueError("Output formants features are INF.")
        if not torch.all(torch.isfinite(tilt)):
            raise ValueError("Output tilt features are INF.")
        if not torch.all(torch.isfinite(centroid)):
            raise ValueError("Output centroid features are INF.")
        if not torch.all(torch.isfinite(energy)):
            raise ValueError("Output energy features are INF.")
        if not torch.all(torch.isfinite(voicing_flag)):
            raise ValueError("Output voicing_flag features are INF.")
        
        x = torch.transpose(torch.cat((pitch.unsqueeze(1), formants, tilt.unsqueeze(1), centroid.unsqueeze(1), energy.unsqueeze(1), voicing_flag.unsqueeze(1)),dim = -1), 0, 1)
        y_mel = mel_spectrogram(audio,sampling_rate = self.sampling_rate, n_fft=self.frame_size, win_size = self.frame_size, hop_size = self.hop_size, fmin = 0.0, fmax = self.config.fmax_for_loss, num_mels = 128)

        if torch.any(torch.isnan(audio)):
            raise ValueError("Output audio features are NaN.")
        
        if x.shape[-1] != self.segment_length:
            raise ValueError("Wrong x length.")
        if audio.shape[-1] != int(self.segment_length * self.hop_size):
            raise ValueError("Wrong audio length.")

        return x.type(torch.FloatTensor).to(self.device), audio.squeeze(0).type(torch.FloatTensor).to(self.device), y_mel.squeeze().type(torch.FloatTensor).to(self.device)