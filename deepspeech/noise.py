import librosa, sox, os
import numpy as np
import scipy.signal
import soundfile as sf
import pandas as pd
from tempfile import NamedTemporaryFile

def load_audio(path):
    sound, sample_rate = sf.read(path, dtype='int16')
    # TODO this should be 32768.0 to get twos-complement range.
    # TODO the difference is negligible but should be fixed for new models.
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        # if not os.path.exists(path):
        #    print("Directory doesn't exist: {}".format(path))
        #    raise IOError
        # self.paths = path is not None and librosa.util.find_files(path)
        self.paths = path
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.abs(np.random.rand() * (noise_len - data_len))
        noise_end = noise_start + data_len
        # noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        noise_dst = audio_with_librosa(noise_path, self.sample_rate, noise_start, noise_end)
        while len(data) > len(noise_dst):
            noise_dst = list(noise_dst)
            noise_dst += noise_dst
        noise_dst = np.array(noise_dst[:len(data)])
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data

def audio_with_librosa(path, sample_rate, start_time, end_time):
    start = int(start_time * sample_rate)
    end = int(end_time * sample_rate)
    wav, sr = librosa.load(path, sr=sample_rate)
    return wav[start:end]

def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate, tar_filename, start_time, end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y

