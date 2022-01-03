import torch
import torch.nn as nn
import json
from decoder import GreedyDecoder, BeamCTCDecoder
from model import DeepSpeech
import soundfile as sf
import librosa
import numpy as np
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def setting(cfg: DictConfig):
    path = cfg.train.MODEL_PATH

    with open(cfg.json_path, encoding='utf-8-sig') as f:
        labels = json.load(f)
    config = cfg.audio
    
    return path, labels, config

def STT(wav_path):
    path, labels, config = setting()
    model = DeepSpeech(rnn_type=nn.GRU, labels=labels, rnn_hidden_size=1024, nb_layers=5, audio_conf=config,
                                bidirectional=True, context=20)
    model.load_state_dict(torch.load(path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    greedy = GreedyDecoder(labels)
    beam = BeamCTCDecoder(labels)

    spec = wav2spec(wav_path, config)

    spec = spec.to(device)
    spec = spec.reshape((1, 1, spec.shape[0], spec.shape[1]))
    sizes = [spec.shape[3]]
    sizes = torch.IntTensor(sizes)

    out, output_sizes = model(spec, sizes)
    sentence, offsets = greedy.decode(out)
    print(sentence)


def wav2spec(path, config):
    sound, sr = sf.read(path)
    sound = sound.astype('float32') / 32767
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)
    n_fft = int(config['sample_rate'] * config['window_size'])
    win_length = n_fft
    hop_length = int(config['sample_rate'] * config['window_stride'])
    D = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=config['window'])
    spec, phase = librosa.magphase(D)
    spec = np.log1p(spec)
    spec = torch.FloatTensor(spec)
    return spec
