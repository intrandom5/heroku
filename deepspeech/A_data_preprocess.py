import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import re
import glob
import csv
import json

@hydra.main(config_path="", config_name="config")
def main(cfg: DictConfig):
    wav_list, txt_list = construct_datalist(cfg.data.data_directory)
    sentences = open_txt_files(txt_list)
    labels = save_label_to_json(sentences, cfg.data.json_path)
    str2int = {}
    for i, label in enumerate(labels):
        str2int[i] = label
    encoded = convert_to_integer(sentences, str2int)
    with open(cfg.data.csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["wav", "txt", "encode"])
        for wav, txt, encode in zip(wav_list, txt_list, encoded):
            wr.writerow([wav, txt, encode])
    
def construct_datalist(directory):
    """
    dataset should be organized like this
    folder_name/
        xxx1.wav
        xxx1.txt
        xxx2.wav
        xxx2.txt
        ...
    """
    wav_list = glob.glob(directory + "/*.wav")
    txt_list = [wav[:-3] + "txt" for wav in wav_list]
    return wav_list, txt_list

def open_txt_files(textfiles):
    results = []
    for txt in textfiles:
        try:
            with open(txt, encoding='utf-8') as f:
                sentence = f.readline()
        except:
            with open(txt, encoding='cp949') as f:
                sentence = f.readline()
        results.append(sentence)
    return results
    
def save_label_to_json(sentences, json_path):
    train_label_list = []
    train_label_freq = []

    for sentence in sentences:
        for ch in sentence:
            if ch not in train_label_list:
                train_label_list.append(ch)
                train_label_freq.append(1)
            else:
                train_label_freq[train_label_list.index(ch)] += 1

    # sort together Using zip
    train_label_freq, train_label_list = zip(*sorted(zip(train_label_freq, train_label_list), reverse=True))
    label = {'id': [], 'char': []}
    for idx, (ch, freq) in enumerate(zip(train_label_list, train_label_freq)):
        label['id'].append(idx)
        label['char'].append(ch)
        
    df = pd.DataFrame(label)
    target = list(df['char'])
    
    with open(json_path, "w", encoding="UTF-8-sig") as f:
        f.write(json.dumps(target, ensure_ascii=False)
    return target
    
def convert_to_integer(sentences, s2i):
    results = []
    for sentence in sentences:
        result = ''
        for ch in sentence:
            result += (str(s2i[ch]) + ' ')
        results.append(result[:-1])
    return results
        
        
if __name__ == "__main__":
    main()
