import argparse
import re
import os

import torch
import pandas as pd

from model.seqlabel import SeqLabel
from utils.data import Data
from utils.alphabet import Alphabet
from main import evaluate


def evaluate_models(models, datas, name="test"):
    pred_words_list = []
    for model, data in zip(models, datas):
        speed, acc, p, r, f, gold_words, pred_words = evaluate(data, model, name)
        pred_words_list.append(pred_words)
        print(speed, acc, p, r, f)
    return gold_words, pred_words_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('-n', type=int, help='num')
    parser.add_argument('-m1', help='model 1 (baseline)')
    parser.add_argument('-d1', help='data 1 (baseline)')
    parser.add_argument('-m2', help='model 2')
    parser.add_argument('-d2', help='data 2')
    parser.add_argument('-m3', help='model 3')
    parser.add_argument('-d3', help='data 3')
    parser.add_argument('-m4', help='model 4')
    parser.add_argument('-d4', help='data 4')
    parser.add_argument('-m5', help='model 5')
    parser.add_argument('-d5', help='data 5')
    parser.add_argument('-m6', help='model 6')
    parser.add_argument('-d6', help='data 6')
    parser.add_argument('--out', default='eval.csv', help='out path')
    args = parser.parse_args()

    datas = []
    models = []
    names = []
    for i in range(args.n):
        data = Data()
        data.load(args.__getattribute__("d{}".format(i + 1)))
        data.HP_gpu = False
        model = SeqLabel(data)
        model.load_state_dict(torch.load(args.__getattribute__("m{}".format(i + 1)), map_location=lambda storage, loc: storage))
        datas.append(data)
        models.append(model)
        names.append(os.path.split(args.__getattribute__("m{}".format(i + 1)))[-1])

    gold_words, pred_words_list = evaluate_models(models, datas)
    all_words = set(gold_words + [pred_word for pred_words in pred_words_list for pred_word in pred_words])
    
    datas[0].word_alphabet = Alphabet('word')
    datas[0].build_alphabet(data.train_dir)
    raws = []

    for word in all_words:
        raw = {"word": re.sub("\d+_\d+_", "", word), "iv_num": 0, "oov_num": 0, "word_num": 0}
        for w in raw["word"].split(" "):
            if w in datas[0].word_alphabet.instances:
                raw["iv_num"] += 1
            else:
                raw["oov_num"] += 1
            raw["word_num"] += 1
        if word in gold_words:
            raw["gold"] = 1
        else:
            raw["gold"] = 0
        for i in range(len(pred_words_list)):
            if word in pred_words_list[i]:
                raw[names[i]] = 1
            else:
                raw[names[i]] = 0
        raws.append(raw)
    
    df = pd.DataFrame(raws)
    df.to_csv(args.out)
