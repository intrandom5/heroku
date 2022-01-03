import torch
import torch.nn as nn
import training
import soundfile as sf
import librosa
import numpy as np
import json
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt
from decoder import GreedyDecoder, BeamCTCDecoder
from evaluation import evaluate
from loader import AudioLoader, AudioDataLoader
from torch.nn import CTCLoss
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="", config_name="config")
def main(cfg: DictConfig):
    CONTINUE = cfg.train.CONTINUE
    SAVE_PATH = cfg.train.MODEL_PATH
    progress_states_path = cfg.train.progress_states_path
    label_path = cfg.data.json_path
    train_csv_path = cfg.data.csv_path

    audio_conf = cfg.audio
        
    spec_aug = cfg.train.SPEC_AUG
    noise_inject = cfg.train.NOISE_INJECT
    batch_size = cfg.train.BATCH_SIZE
    epochs = cfg.train.EPOCHS

    with open(label_path, encoding='utf-8-sig') as f:
        labels = json.load(f)
    train_dataset = AudioLoader(train_csv_path, spec_aug, noise_inject)
    train_loader = AudioDataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size)
    
    model = training.DeepSpeech(rnn_type=nn.GRU, labels=labels, rnn_hidden_size=1024, nb_layers=5, audio_conf=audio_conf,
                                bidirectional=True, context=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=3e-4, momentum=0.9, nesterov=True, weight_decay=1e-5)
    criterion = CTCLoss()
    
    done_epochs = 0
    train_loss = []
    val_loss = []
    iteration = 0
    total_iter = len(train_loader)

    # load model states if exists.
    if CONTINUE:
        with open(progress_states_path, 'rb') as f:
            progress = pickle.load(f)
        done_epochs = progress['epoch']
        iteration = progress['iteration']
        if iteration > 17300:
            done_epochs += 1
            iteration = 0
        train_loss = progress['train loss']
        val_loss = progress['val loss']
        path = progress['model weights']
        model.load_state_dict(torch.load(path))
        print(done_epochs, iteration)
    
    # training
    for epoch in range(done_epochs, epochs):
        print("epochs :", epoch)
        # inputs : 음성 스펙트로그램 (배치 사이즈, 채널(1), 음성시간, 주파수) - 모든 음성 길이를 가장 긴 음성에 맞춤.
        # targets : 라벨링 된 문자열
        # per : 해당 문장 길이/해당 문장이 속한 미니 배치의 가장 긴 문장 길이
        # target_size : 해당 문장 길이
        for i, (inputs, targets, per, target_size) in enumerate(train_loader):
            if iteration != 0 and i < iteration:
                continue
            elif iteration != 0 and i >= iteration:
                print(f"continue from {done_epochs} epoch, {iteration} iteration.")
                iteration = 0

            # input_sizes : 제각각 다른 음성 파일 길이들을 담은 리스트. 이 변수를 추가로 forward 함수에 제공.
            input_sizes = per.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)
            out, output_sizes = model(inputs, input_sizes)
            # CTC Loss 계산을 위해 transpose
            out = out.transpose(0, 1)
            float_out = out.float()
            loss = criterion(float_out, targets, output_sizes, target_size).to(device)
            loss = loss / inputs.size(0)
            loss_value = loss.item()
            if i % 300 == 0 and i != 0:
                print(f"Iter : {i}/{total_iter}, train loss : {loss_value}")
                train_loss.append(loss_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), SAVE_PATH)
            progress = {"epoch": epoch, "iteration": i, "train loss": train_loss, "val loss": val_loss,
                        "model weights": SAVE_PATH}
            with open(progress_states_path, 'wb') as f:
                pickle.dump(progress, f)
        
    plt.plot(train_loss, "r", label="train loss")
    plt.xlabel("300*x iterations")
    plt.ylabel("loss")
    plt.savefig("training_loss_graph.png")


if __name__ == "__main__":
    main()
