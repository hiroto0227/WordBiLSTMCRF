import numpy as np
import re
from nltk.tokenize import word_tokenize

BLANK_REGEX = " | | | | | | | | | |\x09|\x20|\xc2\xa0|\xe2\x80\x82|\xe2\x80\x83|\xe2\x80\x84|\xe2\x80\x85|\xe2\x80\x86|\xe2\x80\x87|\xe2\x80\x88|\xe2\x80\x89|\xe2\x80\x8a|\xe2\x80\x8b|\xe3\x80\x80|\xef\xbb\xbf"

def tokenize(text):
    """textをtoken単位に分割したリストを返す。"""
    tokens = [re.sub("(" + BLANK_REGEX + ")", " ", token) for token in re.split("(" + BLANK_REGEX + "|\xa0|\t|\n|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|-|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|→|¬|®|•|′|°|~|∼|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|™|⋅|-|\u2000|⁺|\u2009|\d+\.\d+|\d{3,}|[Α-Ωα-ω]+)", text)]
    return list(filter(None, tokens))
 
def nltk_tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def load_data(seqfile):
    word_seq, label_seq = [], []
    with open(seqfile, "rt") as f:
        _word, _label = [], []
        for i, line in enumerate(f.read().split("\n")[:-2]):
            splited_line = line.split("\t")
            # \nがくるごとにそれぞれの系列を追加し、初期化(\n)が二連続の場合もある。
            if len(splited_line) <= 1:
                if len(_word) >= 1:
                    word_seq.append(_word)
                    label_seq.append(_label)
                _word, _label, = [], []
            else:
                _word.append(splited_line[0])
                _label.append(splited_line[-1])
    return word_seq, label_seq


def batch_gen(word_seq,
              char_seq,
              tokens_seq,
              f_masks_seq,
              b_masks_seq,
              label_seq,
              batch_size,
              word_pad_ix=0,
              char_pad_ix=0,
              label_pad_ix=0,
              tokens_pad_ix=0,
              shuffle=True):
    word_batch, label_batch, mask_batch, char_batch = [], [], [], []
    f_masks_batch = [[] for i in range(len(f_masks_seq))]
    b_masks_batch = [[] for i in range(len(b_masks_seq))]
    tokens_batch = [[] for i in range(len(tokens_seq))]
    max_len = 0
    char_max_len = 0
    ixs = np.arange(len(word_seq))
    if shuffle:
        np.random.shuffle(ixs)
    for i, ix in enumerate(ixs):
        if max_len < len(word_seq[ix]):
            max_len = len(word_seq[ix])
        for j in range(len(tokens_seq)):
            if max_len < len(tokens_seq[j][ix]):
                max_len = len(tokens_seq[j][ix])
        word_batch.append(word_seq[ix])
        mask_batch.append([1 for _ in range(len(word_seq[ix]))])
        label_batch.append(label_seq[ix])
        char_batch.append(char_seq[ix])
        for j, token_seq in enumerate(tokens_seq):
            tokens_batch[j].append(token_seq[ix])
        for j, f_mask_seq in enumerate(f_masks_seq):
            f_masks_batch[j].append(f_mask_seq[ix])
        for j, b_mask_seq in enumerate(b_masks_seq):
            b_masks_batch[j].append(b_mask_seq[ix])
        if (i + 1) % batch_size == 0:
            padded_chars_batch, char_masks_batch = padding_char(char_batch, max_len, char_pad_ix)
            yield (padding(word_batch, max_len, word_pad_ix),
                   padded_chars_batch,
                   [padding(token_batch, max_len, tokens_pad_ix) for j, token_batch in enumerate(tokens_batch)],
                   [padding(f_mask_batch, max_len, 1) for j, f_mask_batch in enumerate(f_masks_batch)],
                   [padding(b_mask_batch, max_len, 1) for j, b_mask_batch in enumerate(b_masks_batch)],
                   padding(label_batch, max_len, label_pad_ix),
                   padding(mask_batch, max_len, 0),
                   char_masks_batch)
            word_batch, label_batch, mask_batch, char_batch = [], [], [], []
            tokens_batch = [[] for i in range(len(tokens_seq))]
            f_masks_batch = [[] for i in range(len(f_masks_seq))]
            b_masks_batch = [[] for i in range(len(b_masks_seq))]
            max_len = 0
        

def padding(batches, max_len, pad_ix):
    pad_batches = []
    for batch in batches:
        pad_length = max_len - len(batch)
        pad_batches.append(list(batch) + [pad_ix for i in range(pad_length)])
    return pad_batches

def padding_char(char_batch, seq_max_len, char_pad_ix):
    padded_chars_batch = []
    char_masks_batch = []
    # まずはsequenceレベルでpadding
    char_batch = padding(char_batch, seq_max_len, pad_ix=[])
    char_max_len = 0
    for chars_list in char_batch:
        for chars in chars_list:
            if char_max_len < len(chars):
                char_max_len = len(chars)
    for chars_list in char_batch:
        padded_chars = []
        char_masks = []
        for chars in chars_list:
            pad_length = char_max_len - len(chars)
            char_masks.append([1 for i in range(len(chars))] + [0 for i in range(pad_length)])
            padded_chars.append(chars + [char_pad_ix for i in range(pad_length)])
        padded_chars_batch.append(padded_chars)
        char_masks_batch.append(char_masks)
    return padded_chars_batch, char_masks_batch
