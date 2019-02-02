# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-10 15:08:07
from __future__ import print_function
from __future__ import absolute_import
import sys
import re
import numpy as np


cammel_re = re.compile("^[A-Z][a-z0-9]+$")

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def cammel_normalize_word(word):
    if cammel_re.match(word):
        return word.lower()
    return word

def read_instance(input_file, word_alphabet, char_alphabet, sw_alphabet_list, feature_alphabets, label_alphabet, number_normalized, max_sent_length, sentencepieces, char_padding_size=-1, char_padding_symbol = '</pad>'):
    feature_num = len(feature_alphabets)
    sw_num = len(sentencepieces)
    in_lines = open(input_file,'r').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    sws_list = [[] for _ in range(sw_num)]
    sw_fmasks_list = [[] for _ in range(sw_num)]
    sw_bmasks_list = [[] for _ in range(sw_num)]
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    sw_Ids_list = [[] for _ in range(sw_num)]
    label_Ids = []

    ### for sequence labeling data format i.e. CoNLL 2003
    for z, line in enumerate(in_lines):
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if sys.version_info[0] < 3:
                word = word.decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            ## get features
            feat_list = []
            feat_Id = []
            for idx in range(feature_num):
                feat_idx = pairs[idx+1].split(']',1)[-1]
                feat_list.append(feat_idx)
                feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
            features.append(feat_list)
            feature_Ids.append(feat_Id)
            ## get char
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

            ## deal with sw
            for idx, sp in enumerate(sentencepieces):
                sw_list = []
                sw_Id = []
                sw_fmask = []
                sw_bmask = []
                for sw in sp.tokenize(word):
                    if number_normalized:
                        sw = normalize_word(sw)
                    sw_list.append(sw)
                    sw_Id.append(sw_alphabet_list[idx].get_index(sw))
                    sw_fmask.append(0)
                    sw_bmask.append(0)
                sw_fmask[-1] = 1
                sw_bmask[0] = 1
                sw_fmasks_list[idx].extend(sw_fmask)
                sw_bmasks_list[idx].extend(sw_bmask)
                sws_list[idx].extend(sw_list)
                sw_Ids_list[idx].extend(sw_Id)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                instence_texts.append([words, features, chars, sws_list, labels])
                instence_Ids.append([word_Ids, feature_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list, label_Ids])
            # if z < 10000:
            #     print("short cut read_instance!!!")
            #     break
            words = []
            features = []
            chars = []
            sws_list = [[] for _ in range(sw_num)]
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            sw_Ids_list = [[] for _ in range(sw_num)]
            sw_fmasks_list = [[] for _ in range(sw_num)]
            sw_bmasks_list = [[] for _ in range(sw_num)]
            label_Ids = []
    if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
        instence_texts.append([words, features, chars, sws_list, labels])
        instence_Ids.append([word_Ids, feature_Ids, char_Ids, sw_Ids_list, sw_fmasks_list, sw_bmasks_list, label_Ids])
        words = []
        features = []
        chars = []
        sws_list = [[] for _ in range(sw_num)]
        labels = []
        word_Ids = []
        feature_Ids = []
        char_Ids = []
        sw_Ids_list = [[] for _ in range(sw_num)]
        sw_fmasks_list = [[] for _ in range(sw_num)]
        sw_bmasks_list = [[] for _ in range(sw_num)]
        label_Ids = []
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float)
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrain_emb[np.isnan(pretrain_emb)] = np.random.uniform(-scale, scale, [1])
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            #else:
            #     assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            try:
                embedd[:] = tokens[1:]
                if sys.version_info[0] < 3:
                    first_col = tokens[0].decode('utf-8')
                else:
                    first_col = tokens[0]
                embedd_dict[first_col] = embedd
            except ValueError:
                print("Value Error: {}".format(line))
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
