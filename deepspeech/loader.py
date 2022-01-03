from torch.utils.data import Dataset, DataLoader
import librosa, sox, os
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import pandas as pd
import time

from spec_augment import spec_augment
from noise import NoiseInjection

# 데이터로더
class AudioLoader(Dataset) :

    def __init__(self, csv_file, aug=False, noise_inject=False) :
        self.data_df = pd.read_csv(csv_file)
        self.config = dict(sample_rate=16000, window_size=.02, window_stride=.01, window=scipy.signal.hamming)
        self.noise_inject = noise_inject
        self.spec_augment = aug
        if self.noise_inject:
            noise_paths = self.data_df['wav']
            self.injector = NoiseInjection(path=noise_paths, noise_levels=(0, 0.4))
    
    #데이터셋의 길이 반환
    def __len__(self) : 
        return len(self.data_df)

    def load_audio(self, path):
        if path.endswith('pcm'):
            with open(path, 'rb') as f:
                buf = f.read()
                pcm_data = np.frombuffer(buf, dtype='int16')
            sound = librosa.util.buf_to_float(pcm_data, 2)
        else:
            sound, sample_rate = sf.read(path, dtype='int16')
        # TODO this should be 32768.0 to get twos-complement range.
        # TODO the difference is negligible but should be fixed for new models.
        sound = sound.astype('float32') / 32767  # normalize audio
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # multiple channels, average
        if self.noise_inject:
            injected = self.injector.inject_noise(sound)
            return injected
        return sound
    
    def get_spec(self, path):
        audio = self.load_audio(path)
        n_fft = int(self.config["sample_rate"] * self.config["window_size"])
        win_length = n_fft
        hop_length = int(self.config["sample_rate"] * self.config["window_stride"])
        #STFT
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=self.config["window"])
        spec, phase = librosa.magphase(D)
        spec = np.log1p(spec)
        spec = torch.FloatTensor(spec)
        if self.spec_augment:
            spec = spec_augment(spec)
        return spec
    
    # 숫자로 인코딩 결과
    def get_transcript(self, script):
        return [int(code) for code in script.split()]

    
    #데이터셋의 n번째 아이템 반환
    def __getitem__(self, index) : 
        #이미지 목표(레이블)
        label = self.data_df['encode'][index]
        wav = self.data_df['wav'][index]
        inputs = self.get_spec(wav)
        targets = self.get_transcript(label)
        return inputs, targets

# 데이터로더를 미니배치 단위로 반환하고자 할 때 사용자가 설정할 수 있는 커스텀 함수라고 함.
# 여기선 spectrogram, label 외에 input_percentages와 target_sizes를 추가로 반환하도록 설정하고 있음.
# input_percentages = (하나의 음성의 길이)/(미니배치 내에서 가장 긴 음성의 길이)
# target_sizes = (label 수, 문자 개수라고 보면 됨.)
def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


# 배치 단위 데이버 반환을 위한 클래스
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
