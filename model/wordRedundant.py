import sys, os
import backtrace
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
#from torchcrf import CRF

sys.path.append("../")
sys.path.append("./")
from sentencepieces import sp_tokenizer
from utils import data, vocab, trainutils, labels, lossmanager
from utils.trainutils import get_variable
from load_config import load_state
#from bilstmcrf import BiLSTMLSTMCRF
from simple_bilstmcrf import BiLSTMLSTMCRF
from pretrain.pretrain import load_word2vec, make_pretrain_embed


def writeSeqsTSV(word_seq, label_seq, pred_seq, outpath, mode="a"):
    with open(outpath, mode) as f:
        for i in range(len(word_seq)):
            pred_label = labels.O if pred_seq[i] in [labels.UNK, labels.PAD] else pred_seq[i]
            row = [word_seq[i]] + [label_seq[i]] + [pred_label]
            f.write("\t".join(row) + "\n")
        f.write("\n")
    return True


def makeMask(word_seq, token_seq, mode="f"):
    masks = []
    for words, tokens in zip(word_seq, token_seq):
        mask = [0 for i in range(len(tokens))]
        word_ixs, word_ix, token_ix = [], 0, 0
        if mode == "b":
            words = list(reversed(words))
            tokens = list(reversed(tokens))
        for word in words:
            word_ix += len(word)
            word_ixs.append(word_ix)
        for i, token in enumerate(tokens):
            token_ix += len(token)
            if token_ix in word_ixs:
                mask[i] = 1
        if mode == "b":
            mask = list(reversed(mask))
        assert sum(mask) == len(words), "Failed Making Mask. {}, {}".format(sum(mask), len(words))
        masks.append(mask)
    return masks


class WordRedundant:
    def __init__(self, args):
        self.args = args
        self.sentencepieces = []
        if args.sp_models:
            for sp_path in args.sp_models:
                sp = sp_tokenizer.SentencePieceTokenizer()
                print(sp_path)
                sp.load(sp_path)
                self.sentencepieces.append(sp)
        
    def train(self, char=True):
        loss_manager = lossmanager.LossManager()
        word_encoder = vocab.LabelEncoder()
        char_encoder = vocab.LabelEncoder()
        label_encoder = vocab.LabelEncoder(label=True)
        token_encoders = [vocab.LabelEncoder() for i in range(len(self.sentencepieces))]

        word_seq, label_seq = data.load_data(self.args.train_path)
        word_vec = word_encoder.fit_transform(word_seq)
        label_vec = label_encoder.fit_transform(label_seq)
        char_seq = [[list(word) for word in words] for words in word_seq]
        char_encoder.fit([chars for char_list in char_seq for chars in char_list])
        char_vecs = [char_encoder.transform(char_list) for char_list in char_seq]
        token_vecs, f_masks_seq, b_masks_seq = [], [], []
        for i, sp in enumerate(self.sentencepieces):
            # word_seqからtextを構成し、tokenizerでtoken_seqを作成する。
            token_seq = [list(filter(lambda x: x != " ", sp.tokenize(" ".join(words)))) for words in word_seq]
            f_masks_seq.append(makeMask(word_seq, token_seq, mode="f"))
            b_masks_seq.append(makeMask(word_seq, token_seq, mode="b"))
            token_vec = token_encoders[i].fit_transform(token_seq)
            token_vecs.append(token_vec)
        
        # load pretrain embedding
        if self.args.word2vec_path:
            word2vec = load_word2vec(self.args.word2vec_path)
            word_pretrain_embed = make_pretrain_embed(word2vec, word_encoder.label2id, self.args.word_embed_dim)

        token_pretrain_embeds = []
        if self.args.token2vec_path:
            for token2vec_path, token_encoder in zip(self.args.token2vec_path, token_encoders):
                print("===== {}, vocab size={} =====".format(token2vec_path, len(token_encoder.label2id)))
                token2vec = load_word2vec(token2vec_path)
                pretrain_embed = make_pretrain_embed(token2vec, token_encoder.label2id, self.args.token_embed_dim)
                token_pretrain_embeds.append(pretrain_embed)
        
        model = BiLSTMLSTMCRF(word_vocab_dim=len(word_encoder.label2id), 
                              char_vocab_dim=len(char_encoder.label2id),
                              token_vocab_dims=[len(te.label2id) for te in token_encoders],
                              label_dim=len(label_encoder.label2id),
                              token_embed_dim=self.args.token_embed_dim,
                              word_embed_dim=self.args.word_embed_dim,
                              char_embed_dim=self.args.char_embed_dim,
                              word_lstm_dim=self.args.word_lstm_dim,
                              token_lstm_dim=self.args.token_lstm_dim,
                              char_lstm_dim=self.args.char_lstm_dim,
                              word_pretrain_embed=word_pretrain_embed,
                              token_pretrain_embeds=token_pretrain_embeds,
                              dropout=self.args.dropout,
                              training=True,
                              unk_ix=word_encoder.label2id[labels.UNK],
                              zero_ixs=[word_encoder.label2id[labels.PAD]],
                              gpu=self.args.gpu)
        print(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        model.train()

        #beta = (0.9,0.98)
        #optimizer = optim.Adam(model.parameters(), lr=self.args.initial_rate, weight_decay=self.args.weight_decay, betas=beta)
        optimizer = optim.SGD(model.parameters(), lr=self.args.initial_rate, weight_decay=self.args.weight_decay)
        print("seq length: {}".format(len(word_vec)))
        print("label2id: {}".format(label_encoder.label2id))
        
        for epoch in range(1, self.args.epoch + 1):
            print("\n\n================ {} epoch ==================".format(epoch))
            model.train()
            model.zero_grad()
            start = time.time()
            loss_per_epoch = 0
            for i, (word_batch, char_batch, tokens_batch, f_masks_batch, b_masks_batch, label_batch, mask_batch, char_mask_batch) in \
                   tqdm(enumerate(data.batch_gen(word_vec, 
                                                 char_vecs, 
                                                 token_vecs, 
                                                 f_masks_seq, 
                                                 b_masks_seq, 
                                                 label_vec, 
                                                 self.args.batchsize, 
                                                 word_pad_ix=word_encoder.label2id[labels.PAD], 
                                                 label_pad_ix=label_encoder.label2id[labels.PAD], 
                                                 char_pad_ix=char_encoder.label2id[labels.PAD]))):
                word_batch = trainutils.get_variable(torch.LongTensor(word_batch), gpu=self.args.gpu).transpose(1, 0)
                char_batch = trainutils.get_variable(torch.LongTensor(char_batch), gpu=self.args.gpu).transpose(1, 0)
                tokens_batch = [trainutils.get_variable(torch.LongTensor(token_batch), gpu=self.args.gpu).transpose(1, 0) for token_batch in tokens_batch]
                label_batch = trainutils.get_variable(torch.LongTensor(label_batch), gpu=self.args.gpu).transpose(1, 0)
                f_masks_batch = [trainutils.get_variable(torch.LongTensor(f_mask_batch), gpu=self.args.gpu).transpose(1, 0) for f_mask_batch in f_masks_batch]
                b_masks_batch = [trainutils.get_variable(torch.LongTensor(b_mask_batch), gpu=self.args.gpu).transpose(1, 0) for b_mask_batch in b_masks_batch]
                mask_batch = trainutils.get_variable(torch.ByteTensor(mask_batch), gpu=self.args.gpu).transpose(1, 0)
                char_mask_batch = trainutils.get_variable(torch.FloatTensor(char_mask_batch), gpu=self.args.gpu).transpose(1, 0)
                loss = model.loss(word_batch, char_batch, tokens_batch, f_masks_batch, b_masks_batch, label_batch, mask_batch, char_mask_batch)
                if i % 10 == 0:
                    loss_manager.append(loss)
                    print("\nword_batch: {}\nloss: {}".format(word_batch.shape, loss))
                loss.backward()
                optimizer.step()
                model.zero_grad()
                loss_per_epoch += float(loss)
            print("loss_per_epoch: {}\ntime_per_epoch: {}".format(loss_per_epoch, time.time() - start))
            if epoch % 10 == 0:
                torch.save(model.state_dict(), self.args.save_model + str(epoch))
                print("model saved! {}".format(self.args.save_model + str(epoch)))
        loss_manager.draw_graph(self.args.save_model + ".png")
        torch.save(model.state_dict(), self.args.save_model)

    def predict(self):
        if os.path.exists(self.args.predicted_path):
            os.remove(self.args.predicted_path)

        word_encoder = vocab.LabelEncoder()
        char_encoder = vocab.LabelEncoder()
        label_encoder = vocab.LabelEncoder(label=True)
        token_encoders = [vocab.LabelEncoder() for i in range(len(self.sentencepieces))]
        
        word_seq, label_seq = data.load_data(self.args.train_path)
        word_encoder.fit(word_seq)
        label_encoder.fit(label_seq)
        char_seq = [[list(word) for word in words] for words in word_seq]
        char_encoder.fit([chars for char_list in char_seq for chars in char_list])
        for i, sp in enumerate(self.sentencepieces):
            # word_seqからtextを構成し、tokenizerでtoken_seqを作成する
            token_seq = [list(filter(lambda x: x != " ", sp.tokenize(" ".join(words)))) for words in word_seq]
            token_encoders[i].fit(token_seq)

        # load pretrain embedding
        if self.args.word2vec_path:
            word2vec = load_word2vec(self.args.word2vec_path)
            word_pretrain_embed = make_pretrain_embed(word2vec, word_encoder.label2id, self.args.word_embed_dim)

        token_pretrain_embeds = []
        token2vecs = []
        if self.args.token2vec_path:
            for token2vec_path, token_encoder in zip(self.args.token2vec_path, token_encoders):
                print("===== {}, vocab size={} =====".format(token2vec_path, len(token_encoder.label2id)))
                token2vec = load_word2vec(token2vec_path)
                token_pretrain_embeds.append(make_pretrain_embed(token2vec, token_encoder.label2id, self.args.token_embed_dim))
                token2vecs.append(token2vec)
        del word_seq, label_seq

        model = BiLSTMLSTMCRF(word_vocab_dim=len(word_encoder.label2id), 
                              char_vocab_dim=len(char_encoder.label2id),
                              token_vocab_dims=[len(te.label2id) for te in token_encoders],
                              label_dim=len(label_encoder.label2id),
                              token_embed_dim=self.args.token_embed_dim,
                              word_embed_dim=self.args.word_embed_dim,
                              char_embed_dim=self.args.char_embed_dim,
                              word_lstm_dim=self.args.word_lstm_dim,
                              token_lstm_dim=self.args.token_lstm_dim,
                              char_lstm_dim=self.args.char_lstm_dim,
                              word_pretrain_embed=word_pretrain_embed,
                              token_pretrain_embeds=token_pretrain_embeds,
                              dropout=self.args.dropout,
                              training=False,
                              unk_ix=word_encoder.label2id[labels.UNK],
                              zero_ixs=[word_encoder.label2id[labels.PAD]],
                              gpu=-1)
        model.load_state_dict(torch.load(self.args.load_model))
        print(model)
        print("label2id: {}".format(label_encoder.label2id))
        print(model.crf.transitions)
        
        word_seq, label_seq = data.load_data(self.args.test_path)
        char_seq = [[list(word) for word in words] for words in word_seq]
        
        for i in tqdm(range(len(word_seq))):
            model.eval()
            word_vec = word_encoder.transform([word_seq[i]])
            label_vec = label_encoder.transform([label_seq[i]]) 
            char_vec = [char_encoder.transform(char_seq[i])]
            token_vecs, f_masks_seq, b_masks_seq = [], [], []
            token_seq = []
            for j, sp in enumerate(self.sentencepieces):
                token_seq = list(filter(lambda x: x != " ", sp.tokenize(" ".join(word_seq[i]))))
                token_vecs.append(token_encoders[j].transform([token_seq]))
                f_masks_seq.append(makeMask([word_seq[i]], [token_seq], mode="f"))
                b_masks_seq.append(makeMask([word_seq[i]], [token_seq], mode="b"))
            char_batch, char_mask_batch = data.padding_char(char_vec, len(word_vec), char_pad_ix=char_encoder.label2id[labels.PAD])
            word_batch = torch.LongTensor(word_vec).transpose(1, 0)
            char_batch = torch.LongTensor(char_batch).transpose(1, 0)
            tokens_batch = [torch.LongTensor(token_vec).transpose(1, 0) for token_vec in token_vecs]
            f_masks_batch = [torch.LongTensor(f_mask).transpose(1, 0) for f_mask in f_masks_seq]
            b_masks_batch = [torch.LongTensor(b_mask).transpose(1, 0) for b_mask in b_masks_seq]
            mask_batch = torch.ByteTensor([[1 for _ in range(len(word_seq[i]))]]).transpose(1, 0)
            char_mask_batch = torch.FloatTensor(char_mask_batch).transpose(1, 0)
            pred_label_ids = model(word_batch, char_batch, tokens_batch, f_masks_batch, b_masks_batch, mask_batch, char_mask_batch, word_seq, word2vec, token_seq, token2vecs)
            #print(pred_label_ids)
            pred_seq = label_encoder.inverse_transform(pred_label_ids)[0]
            
            writeSeqsTSV(word_seq[i], label_seq[i], pred_seq, outpath=self.args.predicted_path, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("word redundant training")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], help="config data path.")
    parser.add_argument("--config-path", type=str, help="config data path.")
    parser.add_argument("--word2vec-path", type=str, help="trained word2vec path.")
    parser.add_argument("--token2vec-path", type=list, nargs='+', help="trained token2vec paths.")
    parser.add_argument("--sp-models", type=list, nargs='+', help="sentence piece model path")
    opt = parser.parse_args()
    args = load_state(opt.config_path, opt)
    if args.token2vec_path:
        args.token2vec_path = ["".join(path) for path in args.token2vec_path]
    
    word_redundant = WordRedundant(args)
    if opt.mode == "train":
        word_redundant.train()
    elif opt.mode == "predict":
        word_redundant.predict()
