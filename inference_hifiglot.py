import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import argparse
import json
import torch
from neural_formant_synthesis.third_party.hifi_gan.env import AttrDict
from neural_formant_synthesis.feature_extraction import feature_extractor, Normaliser, MedianPool1d
from neural_formant_synthesis.models import SourceFilterFormantSynthesisGenerator

import torchaudio as ta
from tqdm import tqdm
from glob import glob
from safetensors.torch import load_file


torch.backends.cudnn.benchmark = True


def generate_wave_list(file_list, a, h, fm_h):

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    target_sr = h.sampling_rate
    win_size = h.win_size
    hop_size = h.hop_size

    feat_extractor = feature_extractor(sr = target_sr,window_samples = win_size, step_samples = hop_size, formant_ceiling = 10000, max_formants = 4)
    median_filter = MedianPool1d(kernel_size = 7, stride = 1, padding = 0, same = True)
    normalise_features = Normaliser()

    generator = SourceFilterFormantSynthesisGenerator(
        fm_config=fm_h,
        g_config=h,
        pretrained_fm=None,
        freeze_fm=False)

    state_dict = load_file(a.checkpoint_path)
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()

    for file in tqdm(file_list, total = len(file_list)):
        x, sample_rate = ta.load(file)
        x = x[0:1].type(torch.DoubleTensor)
        x = ta.functional.resample(x, sample_rate, target_sr)

        formants, energy, centroid, tilt, pitch, voicing_flag = feat_extractor(x.squeeze(0))
        formants = median_filter(formants.T.unsqueeze(1)).squeeze(1).T
        pitch = pitch.squeeze(0)
        voicing_flag = voicing_flag.squeeze(0)

        for i in range(voicing_flag.size(0)):
            if voicing_flag[i] == 1:
                formants[i,0] = formants[i,0] * a.scale_list[0]
                formants[i,1] = formants[i,1] * a.scale_list[1]
                formants[i,2] = formants[i,2] * a.scale_list[2]
        linear_pitch = pitch

        pitch, formants, tilt, centroid, energy = normalise_features(pitch, formants, tilt, centroid, energy)
        norm_feat = torch.transpose(torch.cat((pitch.unsqueeze(1), formants, tilt.unsqueeze(1), centroid.unsqueeze(1), energy.unsqueeze(1), voicing_flag.unsqueeze(1)),dim = -1), 0, 1)
        norm_feat = norm_feat.type(torch.FloatTensor).unsqueeze(0).to(device)

        y_g_hat, _ = generator(norm_feat, linear_pitch.unsqueeze(0).to(device))

        output_file = os.path.splitext(os.path.basename(file))[0] + '_wave_' + str(a.scale_list[0]) + '_' + str(a.scale_list[1]) + '_' + str(a.scale_list[2]) + '.wav'
        output_orig = os.path.splitext(os.path.basename(file))[0] + '_orig.wav'
        out_path = os.path.join(a.output_path, output_file)
        out_orig_path = os.path.join(a.output_path, output_orig)

        ta.save(out_path, y_g_hat.detach().cpu().squeeze(0), target_sr)
        if not os.path.exists(out_orig_path):
            ta.save(out_orig_path, x.type(torch.FloatTensor), target_sr)

def str_to_list(in_str):
    return list(map(float, in_str.strip('[]').split(',')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help="Path to directory containing files to process.", required=True)
    parser.add_argument('--output_path', help="Path to directory to save processed files", required=True)
    parser.add_argument('--config', help="Path to HiFi-GAN config json file", required=True)
    parser.add_argument('--fm_config', help="Path to feature mapping model config json file", required=True)
    parser.add_argument('--audio_ext', default = '.wav', help="Extension of the audio files to process")
    parser.add_argument('--checkpoint_path', help="Path to pre-trained HiFi-GAN model", required=True)
    parser.add_argument('--scale_list', type=str_to_list, default=[1.0, 1.0, 1.0], required=True)

    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    with open(a.fm_config) as f:
        data = f.read()
    json_fm_config = json.loads(data)
    fm_h = AttrDict(json_fm_config)

    if a.input_path is not None:
        file_list = glob(os.path.join(a.input_path,'*' + a.audio_ext))
    else:
        raise ValueError('Input arguments should include either input_path or file_list')

    if not os.path.exists(a.output_path):
        os.makedirs(a.output_path, exist_ok=True)
    torch.manual_seed(h.seed)

    generate_wave_list(file_list, a, h, fm_h)


if __name__ == '__main__':
    main()