import torch
import torch.nn as nn
import training
import soundfile as sf
import librosa
import numpy as np
import json, scipy
from decoder import GreedyDecoder, BeamCTCDecoder
from evaluation import evaluate
from loader import AudioLoader, AudioDataLoader
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="", config_name="config")
def evaluate(cfg: DictConfig):
    path = cfg.train.MODEL_PATH

    with open(cfg.json_path, encoding='utf-8-sig') as f:
        labels = json.load(f)
    config = cfg.audio

    model = training.DeepSpeech(rnn_type=nn.GRU, labels=labels, rnn_hidden_size=1024, nb_layers=5, audio_conf=config,
                                bidirectional=True, context=20)
    model.load_state_dict(torch.load(path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    greedy = GreedyDecoder(labels)
    beam = BeamCTCDecoder(labels)

    test_csv_path = cfg.evaluate.test_csv_path
    batch_size = cfg.evaluate.BATCH_SIZE

    val_dataset = AudioLoader(test_csv_path, False, False)
    val_loader = AudioDataLoader(dataset=val_dataset, num_workers=0, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wer, cer, data = evaluate(val_loader, device, model, beam, greedy, save_output=None, verbose=False, half=False)
    print("wer :", wer)
    print("cer :", cer)
    
if __name__ == "__main__":
    evaluate()

