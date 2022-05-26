import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
from utils_mine.dataset import build_dataset
from tqdm import tqdm
import copy
from utils_mine.bleu_eval import count_score
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from model import build


def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(1, True)

    save_file_best = torch.load('./cache/best_save.data')

    model.load_state_dict(save_file_best['para'])
    model.eval()
    results = []
    references = []
    sources = []
    best_bleu = save_file_best['best_bleu']
    print(best_bleu)
    for i, (batch_src, batch_tar) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            inputs_outputs = config.tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True)
            src_ids = inputs_outputs['input_ids'].to(config.device)
            for bid in range(len(src_ids)):
                outputs = model.generate(src_ids[bid].unsqueeze(0),
                                         max_length=min(512, 2*src_ids[bid].shape[0]),
                                         eos_token_id=config.tokenizer.eos_token_id,
                                         pad_token_id=config.tokenizer.pad_token_id,
                                         bos_token_id=config.tokenizer.bos_token_id,
                                         top_k=1)
                result = [config.tokenizer.decode(u) for u in outputs]
                print(result[0])
                result = [x.replace(' ', '') for x in result]
                result = [x.replace('<pad>', '') for x in result]
                result = [x.replace('▁', '') for x in result]
                result = [x.split('</s>')[0] for x in result]
                print(result[0])
                print("----------------")
                results += result
            references += batch_tar
            sources += batch_src

    result_final = {'srcs': sources, 'prds': results, 'tars': references}
    with open('./data/test/my_results_bart.pkl', 'wb') as f:
        pickle.dump(result_final, f)

    refs = references[:]
    hyps = results[:]
    refs = [[u] for u in refs]
    bleu = count_score(hyps, refs, config)
    print(bleu)


if __name__ == '__main__':
    main()