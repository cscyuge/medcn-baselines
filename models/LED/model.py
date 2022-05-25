import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from importlib import import_module
from tqdm import tqdm
import copy
from utils_mine.bleu_eval import count_score
from utils_mine.dataset import build_dataset, build_iterator
from transformers import LEDForConditionalGeneration, LEDConfig

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'


def build(batch_size, cuda):

    x = import_module('config')
    pretrained_model = 'fnlp/bart-base-chinese'
    config = x.Config(batch_size, pretrained_model)
    train_data = build_dataset(config, './data/train/src_txts.pkl', './data/train/tar_txts.pkl')
    test_data = build_dataset(config, './data/test/src_txts.pkl', './data/test/tar_txts.pkl')
    val_data = build_dataset(config, './data/valid/src_txts.pkl', './data/valid/tar_txts.pkl')
    train_dataloader = build_iterator(train_data, config.batch_size, config.device)
    val_dataloader = build_iterator(val_data, config.batch_size, config.device)
    test_dataloader = build_iterator(test_data, config.batch_size, config.device)

    model_config = LEDConfig(max_encoder_position_embeddings = config.pad_size,
                             max_decoder_position_embeddings = config.pad_size,
                             decoder_start_token_id = config.tokenizer.cls_token_id,
                             pad_token_id = config.tokenizer.pad_token_id,
                             bos_token_id = config.tokenizer.bos_token_id,
                             eos_token_id = config.tokenizer.eos_token_id,
                             use_cache=False)
    model = LEDForConditionalGeneration(model_config)
    model = model.to(config.device)
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.learning_rate
    )

    return model, optimizer, train_dataloader, val_dataloader, test_dataloader, config


def eval_set(model, dataloader, config):
    model.eval()
    results = []
    references = []

    for i, (batch_src, batch_tar) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            max_length = len(batch_tar[0])+3
            inputs_outputs = config.tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True)
            src_ids = inputs_outputs['input_ids'].to(config.device)
            inputs_outputs = config.tokenizer(batch_tar, return_tensors="pt", padding=True, truncation=True)
            tar_ids = inputs_outputs['input_ids'].to(config.device)
            outputs = model(src_ids, labels=tar_ids)
            logits_ = outputs.logits
            _, predictions = torch.max(logits_, dim=-1)
            result = config.tokenizer.batch_decode(predictions)
            result = [config.tokenizer.convert_tokens_to_string(x) for x in result]
            result = [x.replace(' ', '') for x in result]
            result = [x.replace('[PAD]', '') for x in result]
            result = [x.replace('[CLS]', '') for x in result]
            result = [x.split('[SEP]')[0] for x in result]
            results += result
            references += batch_tar
    references = [[u] for u in references]
    tmp = copy.deepcopy(references)
    bleu = count_score(results, tmp, config)
    del tmp

    sentences = []
    for words in results:
        tmp = ''
        for word in words:
            tmp += word
        sentences.append(tmp)
    model.train()
    return sentences, bleu


def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config):
    #training steps
    max_bleu = -99999
    save_file = {}
    # eval_set(model, val_dataloader, config)
    for e in range(config.num_epochs):
        model.train()
        for i, (batch_src, batch_tar) in tqdm(enumerate(train_dataloader)):
            inputs_outputs = config.tokenizer(batch_src+batch_tar, return_tensors="pt", padding=True, truncation=True)
            src_ids = inputs_outputs['input_ids'][0:len(batch_src)].to(config.device)
            tar_ids = inputs_outputs['input_ids'][len(batch_src):].to(config.device)
            model_outputs = model(input_ids=src_ids, labels=tar_ids)

            loss = model_outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                print('train loss:%f' % loss.item())

            # validation steps
        if e >= 0:
            val_results, bleu = eval_set(model, val_dataloader, config)
            print(val_results[0:5])
            print('BLEU:%f' % (bleu))
            if bleu > max_bleu:
                max_bleu = bleu
                save_file['epoch'] = e + 1
                save_file['para'] = model.state_dict()
                save_file['best_bleu'] = bleu
                torch.save(save_file, './cache/best_save.data')
            if bleu < max_bleu - 0.6:
                print('Early Stop')
                break
            print(save_file['epoch'] - 1)

    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val BLEU:%f' % (save_file_best['best_bleu']))
    model.load_state_dict(save_file_best['para'])
    test_results, bleu = eval_set(model, test_dataloader, config)
    print('Test BLEU:%f' % (bleu))
    with open('./result/best_save_bert.out.txt', 'w', encoding="utf-8") as f:
        f.writelines([x + '\n' for x in test_results])
    return bleu
def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(1, True)
    bleu = train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config)
    # print('finish')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
