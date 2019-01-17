import torch.nn as nn
from crf import CRF
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
from utils.trainutils import get_variable


class BiLSTMLSTMCRF(nn.Module):

    def __init__(self, 
                 word_vocab_dim,
                 char_vocab_dim, 
                 token_vocab_dims, 
                 label_dim, 
                 word_embed_dim, 
                 char_embed_dim,
                 token_embed_dim, 
                 word_lstm_dim, 
                 char_lstm_dim,
                 token_lstm_dim, 
                 dropout, 
                 word_pretrain_embed, 
                 token_pretrain_embeds, 
                 training, 
                 unk_ix=None, 
                 zero_ixs=[], 
                 gpu=-1):
        super().__init__()
        self.training = training
        self.gpu = gpu
        self.word_lstm_dim = word_lstm_dim
        self.token_lstm_dim = token_lstm_dim
        self.num_token_layer = len(token_vocab_dims)
        self.unk_ix = unk_ix
        self.word_vocab_dim = word_vocab_dim
        self.token_vocab_dims = token_vocab_dims
        self.char_lstm_dim = char_lstm_dim
        unk_embed_scale = 3
        
        # init word pretrain embedding
        self.word_embedding = nn.Embedding(word_vocab_dim, word_embed_dim)
        self.char_embedding = nn.Embedding(char_vocab_dim, char_embed_dim)
          
        self.char_f_lstm = nn.LSTMCell(char_embed_dim, char_lstm_dim)
        self.char_b_lstm = nn.LSTMCell(char_embed_dim, char_lstm_dim)
        self.word_lstm = nn.LSTM(word_embed_dim + token_lstm_dim * self.num_token_layer * 2 + char_lstm_dim * 2, self.word_lstm_dim, bidirectional=True)
        self.droplstm = nn.Dropout(dropout)
        self.tanh = nn.Linear(self.word_lstm_dim * 2, label_dim + 2)
        self.crf = CRF(label_dim, self.gpu)
        
        if gpu > 0:
            self.char_embedding.cuda(self.gpu)
            self.char_f_lstm.cuda(self.gpu)
            self.char_b_lstm.cuda(self.gpu)
            self.word_embedding.cuda(self.gpu)
            self.droplstm.cuda(self.gpu)
            self.word_lstm.cuda(self.gpu)
            self.tanh.cuda(self.gpu)
            print("=========== use GPU =============")

        print("token_vocab_dims: {}".format(self.token_vocab_dims))

    def init_hidden(self, batch_size, hidden_dim):
        return (get_variable(torch.zeros(2, batch_size, hidden_dim), gpu=self.gpu),
                get_variable(torch.zeros(2, batch_size, hidden_dim), gpu=self.gpu))

    def _set_lstm_cell_state(self, hidden, pre_hidden, mask):
        """LSTM cellのためのhiddenを用意する
        maskが1のところにはhiddenが入り、maskが0のところにはpre_hiddenが入る。
        """
        def Where(cond, h1, h2):
            return (cond * h1) + ((1-cond) * h2)
        return (Where(mask, hidden[0], pre_hidden[0]),
                Where(mask, hidden[1], pre_hidden[1]))
    
    def _char_lstm(self, char, char_mask):
        # Char Embedding
        char_embed = self.char_embedding(char)
        sh = char_embed.shape
        char_embed = char_embed.view(sh[0]*sh[1], sh[2], sh[3]).transpose(1, 0)
        # Char LSTM
        char_mask = char_mask.contiguous().view(sh[0]*sh[1], sh[2]).unsqueeze(2)
        char_mask = char_mask.expand(char_mask.shape[0], char_mask.shape[1], self.char_lstm_dim).transpose(1, 0)
        char_f_hidden, char_b_hidden = self.init_hidden(char_embed.shape[1], self.char_lstm_dim)
        for i in range(char_embed.shape[0]):
            pre_f_hidden = char_f_hidden
            pre_b_hidden = char_b_hidden
            char_f_hidden = self.char_f_lstm(char_embed[i], char_f_hidden)
            char_b_hidden = self.char_b_lstm(char_embed[char_embed.shape[0] - i - 1], char_b_hidden)
            ##################### mask処理 #########################
            char_f_out = char_f_hidden[0]
            char_b_out = char_b_hidden[0]
            char_f_hidden = self._set_lstm_cell_state(char_f_hidden, pre_f_hidden, char_mask[i])
            char_b_hidden = self._set_lstm_cell_state(char_b_hidden, pre_b_hidden, char_mask[i])
        char_out = torch.cat([char_f_out] + [char_b_out], dim=0)
        char_out = char_out.view(sh[0], sh[1],  self.char_lstm_dim * 2)
        return char_out

    def _forward(self, word, char, tokens, char_mask, f_masks, b_masks, word_seq=None, word2vec=None, tokens_seq=None, token2vecs=None):
        self.zero_grad()
        if not self.training:
            self.eval()
        
        batch_size = word.shape[1]
        self.word_lstm_hidden = self.init_hidden(batch_size, self.word_lstm_dim)
        char_out = self._char_lstm(char, char_mask)
        word_embed = self.word_embedding(word)
        word_embed = get_variable(torch.cat([word_embed, char_out], dim=2), gpu=self.gpu)
        lstm_out, self.word_lstm_hidden = self.word_lstm(word_embed, self.word_lstm_hidden)  # (seq_length, bs, word_hidden_dim)
        lstm_out = self.droplstm(lstm_out)
        out = self.tanh(lstm_out)  # (seq_length, bs, tag_dim)
        return out

    def loss(self, word, char, tokens, f_masks, b_masks, label, mask, char_mask):
        out = self._forward(word, char, tokens,  char_mask, f_masks, b_masks)
        # log_likelihoodを最大にすれば良いが、最小化するので-1をかけている。
        #log_likelihood = self.crf.neg_log_likelihood_loss(out, mask, label)
        # print("\n\n=============== word ===================")
        # print(word.transpose(1, 0)[0])
        # print("=============== out ===================")
        # print(out.transpose(1, 0)[0])
        # print("=============== mask ===================")
        # print(mask.transpose(1, 0)[0])
        # print("=============== label ===================")
        # print(label.transpose(1, 0)[0])
        log_likelihood = self.crf.neg_log_likelihood_loss(out.transpose(1, 0), mask.transpose(1, 0), label.transpose(1, 0))
        #print(self.crf.transitions)  #CRFの中身を見れるよ
        return log_likelihood

    def forward(self, word, char, tokens, f_masks, b_masks, mask, char_mask, word_seq=None, word2vec=None, tokens_seq=None, token2vecs=None):
        out = self._forward(word, char, tokens, char_mask, f_masks, b_masks, word_seq=word_seq, word2vec=word2vec, tokens_seq=tokens_seq, token2vecs=token2vecs)
        #decoded = torch.LongTensor(self.crf.decode(out))
        #_, decoded = self.crf._viterbi_decode(out, mask)
        #print(out.transpose(1, 0)[0])
        #print(mask.transpose(1, 0)[0])
        _, decoded = self.crf._viterbi_decode(out.transpose(1, 0), mask.transpose(1, 0))
        return decoded
