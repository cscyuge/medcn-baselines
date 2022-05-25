import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import MT5Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os


def tokenize(tokenizer, txt):
    src = re.sub('\*\*', '', txt).lower()
    tokens = tokenizer.tokenize(src)
    tokens = ['▁'] + tokens + ['</s>']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


def preprocess(path_from, path_to):
    with open(os.path.join(path_from,'dataset-aligned-para.pkl'), 'rb') as f:
        dataset_aligned = pickle.load(f)
    src_all, tar_all = [], []
    for data in dataset_aligned:
        if len(data[0]) > 2 and len(data[1]) > 2:
            src_all.append(data[0])
            tar_all.append(data[1])
    with open(os.path.join(path_to, 'src_txts.pkl'), 'wb') as f:
        pickle.dump(src_all, f)
    with open(os.path.join(path_to, 'tar_txts.pkl'), 'wb') as f:
        pickle.dump(tar_all, f)


def main():
    print('train dataset:')
    preprocess('../../data/train', './data/train')
    print('test dataset:')
    preprocess('../../data/test', './data/test')
    print('valid dataset:')
    preprocess('../../data/valid', './data/valid')

if __name__ == '__main__':
    main()

    # bert_model = 'google/mt5-small'
    # tokenizer = MT5Tokenizer.from_pretrained(bert_model)
    # print(tokenizer.pad_token)
    # print(tokenizer.pad_token_id)




