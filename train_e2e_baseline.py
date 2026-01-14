import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy import signal
import typing
from typing import List
from collections import namedtuple
import math
import functools
from torch import nn
from librosa.filters import mel as librosa_mel_fn
from accelerate import Accelerator

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neural_formant_synthesis.third_party.hifi_gan.env import AttrDict, build_env
from neural_formant_synthesis.third_party.hifi_gan.models import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_adversarial_loss, discriminator_loss

from neural_formant_synthesis.dataset import FeatureDataset_Baseline
from neural_formant_synthesis.third_party.hifi_gan.models import GeneratorBaseline
from neural_formant_synthesis.glotnet.model.feedforward.wavenet import WaveNet

from accelerate import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

torch.backends.cudnn.benchmark = True

class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(
                x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(
                y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))

            loss = loss + self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss = loss + self.mag_weight * self.loss_fn(x_logmels, y_logmels)

        return loss


def train(a, h, fm_h):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(
        project_dir=a.checkpoint_path, logging_dir=a.checkpoint_path
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        os.makedirs(project_config.project_dir, exist_ok=True)
        os.makedirs(project_config.logging_dir, exist_ok=True)
    with accelerator.main_process_first():
        accelerator.init_trackers("log")

    feature_mapping = WaveNet(
        input_channels = fm_h.n_feat,
        output_channels = fm_h.n_out,
        residual_channels = fm_h.res_channels,
        skip_channels = fm_h.skip_channels,
        kernel_size = fm_h.kernel_size,
        dilations = fm_h.dilations,
        causal = fm_h.causal
    )
    generator = GeneratorBaseline(h = h, input_channels = fm_h.n_out)

    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    steps = 0
    last_epoch = -1

    optim_m = torch.optim.AdamW(feature_mapping.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    scheduler_m = torch.optim.lr_scheduler.ExponentialLR(optim_m, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    trainset = FeatureDataset_Baseline(h, sampling_rate = h.sampling_rate, 
                                   frame_size = h.win_size, hop_size = h.hop_size, shuffle = True, audio_ext = '.wav', 
                                   segment_length = h.segment_size // h.hop_size)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    train_loader, feature_mapping, generator, msd, mpd, optim_m, optim_g, optim_d, scheduler_m, scheduler_g, scheduler_d = accelerator.prepare(
        train_loader, feature_mapping, generator, msd, mpd, optim_m, optim_g, optim_d, scheduler_m, scheduler_g, scheduler_d
    )
    
    device = accelerator.device
    msstftloss = MultiScaleMelSpectrogramLoss(h.sampling_rate).to(device)
    
    if a.resume_path != None:
        accelerator.load_state(a.resume_path)
        steps = int(a.resume_path.split("/")[-1].split("-")[-1])
        last_epoch = int(a.resume_path.split("/")[-1].split("_")[0].split("-")[-1])
        
    generator.train()
    mpd.train()
    msd.train()
    
    for epoch in range(max(0, last_epoch), a.training_epochs):
        for i, batch in tqdm(enumerate(train_loader)):
            x, y, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_mel_hat = feature_mapping(x)
            y_g_hat = generator(y_mel)

            optim_m.zero_grad()
            loss_mel_fm = F.l1_loss(y_mel, y_mel_hat)
            accelerator.backward(loss_mel_fm)
            optim_m.step()

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            accelerator.backward(loss_disc_all)
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            loss_gen_all = 0.0
            
            # Adversarial losses
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_adversarial_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_adversarial_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_all + loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f
                
            loss_mel = msstftloss(y, y_g_hat) * 15
            loss_gen_all = loss_gen_all + loss_mel

            accelerator.backward(loss_gen_all)
            optim_g.step()

            if accelerator.is_main_process:
                # checkpointing
                if steps % a.checkpoint_interval == 0:
                    path = os.path.join(
                        a.checkpoint_path,
                        "epoch-{:04d}_step-{:07d}".format(
                            epoch, steps
                        ),
                    )
                    accelerator.save_state(path)

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    accelerator.log({"training/gen_loss_total": loss_gen_all}, step=steps)
                    accelerator.log({"training/mel_feature_loss": loss_mel_fm}, step=steps)
                    accelerator.log({"training/mel_loss": loss_mel}, step=steps)
                    
                    accelerator.log({"training_gan/disc_f_r": sum(losses_disc_f_r)}, step=steps)
                    accelerator.log({"training_gan/disc_f_g": sum(losses_disc_f_g)}, step=steps)
                    accelerator.log({"training_gan/disc_s_r": sum(losses_disc_s_r)}, step=steps)
                    accelerator.log({"training_gan/disc_s_g": sum(losses_disc_s_g)}, step=steps)
                    
                    accelerator.log({"training_gan/gen_f": sum(losses_gen_f)}, step=steps)
                    accelerator.log({"training_gan/gen_s": sum(losses_gen_s)}, step=steps)
                    
                    accelerator.log({"training_gan/loss_fm_f": loss_fm_f}, step=steps)
                    accelerator.log({"training_gan/loss_fm_s": loss_fm_s}, step=steps)
            steps += 1

        scheduler_m.step()
        scheduler_g.step()
        scheduler_d.step()

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--resume_path')
    parser.add_argument('--config', required=True)
    parser.add_argument('--fm_config', required=True)
    parser.add_argument('--training_epochs', default=1000000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval', default=1, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    with open(a.fm_config) as f:
        data = f.read()
    json_fm_config = json.loads(data)
    fm_h = AttrDict(json_fm_config)

    build_env(a.config, 'config.json', a.checkpoint_path)
    h.seed = int(datetime.now().timestamp())
    torch.manual_seed(h.seed)
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.cuda.manual_seed(h.seed)

    train(a, h, fm_h)


if __name__ == '__main__':
    main()
