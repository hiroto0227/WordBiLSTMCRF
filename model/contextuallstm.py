import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextualLSTM(nn.Module):
    def __init__(self, alphabet_size, pretrain_embedding, embedding_dim, hidden_dim, dropout, gpu):
        super().__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim // 2
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_embedding is not None:
            #pretrain_embedding[np.isnan(pretrain_embedding)] = 0
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        self.f_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.b_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        if self.gpu:
            self.drop.cuda()
            self.embeddings.cuda()
            self.f_lstm.cuda()
            self.b_lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb
    
    def get_masked_hidden(self, inputs, seq_length, fmask, bmask, out_seq_length):
        """
            input:
                inputs: Variable(batch_size, seq_length)
                seq_length: numpy array (batch_size, 1)
                output_masks: numpy array (batch_size, seq_length)
            output:
                Variable(batch_size, masked_length, hidden_dim)
        """
        # #### for debug
        # self.inputs = inputs
        # self.seq_length = seq_length
        # self.fmask = fmask
        # self.bmask = bmask
        # self.out_seq_length = out_seq_length
        # #### for debug

        batch_size = int(inputs.size(0))
        sw_seq_len = int(inputs.size(1))
        
        ################ Foward LSTM ####################
        f_hidden = None
        embeds = self.drop(self.embeddings(inputs))
        # forward lstm
        pack_input = pack_padded_sequence(embeds, seq_length, True)
        f_lstm_out, f_hidden = self.f_lstm(pack_input, f_hidden)
        f_lstm_out, _ = pad_packed_sequence(f_lstm_out)
        f_lstm_out = f_lstm_out.transpose(1, 0)
        # mask process
        fmask = fmask.unsqueeze(2).expand(batch_size, fmask.shape[1], self.hidden_dim).byte()
        fmasked_out = f_lstm_out.masked_select(fmask).view(batch_size, out_seq_length, self.hidden_dim)
        
        ############### Backward LSTM ####################
        b_hidden = None
        # reverse for input
        reverse_inputs, reverse_bmasks = [], []
        max_seq_len = seq_length.max()
        for batch_i in range(batch_size):
            reverse_idx = torch.LongTensor([idx for idx in range(seq_length[batch_i]-1, -1, -1)] + list(range(seq_length[batch_i], max_seq_len)))
            if self.gpu:
                reverse_idx = reverse_idx.cuda()
            reverse_inputs.append(inputs[batch_i].index_select(0, reverse_idx))
            reverse_bmasks.append(bmask[batch_i].index_select(0, reverse_idx))
        reverse_inputs_tensor = torch.cat(reverse_inputs).long().view(batch_size, max_seq_len)
        reverse_bmask_tensor = torch.cat(reverse_bmasks).byte().view(batch_size, max_seq_len)
        if self.gpu:
            reverse_inputs_tensor = reverse_inputs_tensor.cuda()
            reverse_bamsk_tensor = reverse_bmask_tensor.cuda()
        # Backward LSTM
        embeds = self.drop(self.embeddings(reverse_inputs_tensor))
        pack_input = pack_padded_sequence(embeds, seq_length, True)
        b_lstm_out, b_hidden = self.b_lstm(pack_input, b_hidden)
        b_lstm_out, _ = pad_packed_sequence(b_lstm_out)
        b_lstm_out = b_lstm_out.transpose(1, 0)
        # mask process
        # 1番目のものよりも他のものがcut後大きくなる場合はそれに合わせる。
        # if bmask[0].sum() < bmask.sum(dim=1).max():
        #     padded_f_lstm_out = torch.zero(batch_size, bmask.sum(dim=1).max(), self.hidden_dim).long()
        #     padded_f_lstm_out[:, :f_lstm_out.shape[1], :] = f_lstm_out
        #     padded_bmask = torch.zero(batch_size, bmask.sum(dim=1).max()).byte()
        #     padded_bmask[:, :bmask.shape[1]] = bmask
        #     f_lstm_out = padded_f_lstm_out
        #     bmask = padded_bmask
        bmask = bmask.unsqueeze(2).expand(batch_size, inputs.shape[1], self.hidden_dim).byte()
        bmasked_out = b_lstm_out.masked_select(bmask).view(batch_size, out_seq_length, self.hidden_dim)
        # output reverse
        bmasked_outs = []
        for batch_i in range(batch_size):
            reverse_idx = torch.LongTensor([idx for idx in range(out_seq_length-1, max(out_seq_length-seq_length[batch_i], 0)-1, -1)] + list(range(0, out_seq_length-seq_length[batch_i])))
            if self.gpu:
                reverse_idx = reverse_idx.cuda()
            bmasked_outs.append(bmasked_out[batch_i].index_select(0, reverse_idx))
        bmasked_out = torch.cat(bmasked_outs).view(batch_size, out_seq_length, self.hidden_dim)
        if self.gpu:
            return torch.cat([fmasked_out, bmasked_out], dim=2).cuda()
        return torch.cat([fmasked_out, bmasked_out], dim=2)

    def get_last_hidden(self, inputs, seq_length):
        """
            input:
                inputs: Variable(batch_size, seq_length)
                seq_length: numpy array (batch_size, 1)
            output:
                Variable(batch_size, hidden_dim)
        """
        batch_size = input.size(0)
        embeds = self.drop(self.embeddings(inputs))
        hidden = None
        pack_input = pack_padded_sequence(embeds, seq_length, True)
        lstm_out, hidden = self.lstm(pack_input, hidden)
        return hidden[0].transpose(1,0).contiguous().view(batch_size,-1)  #(batch_size, -1)
        
    def forward(self, inputs, seq_length, fmask, bmask):
        return self.get_masked_hidden(inputs, seq_length, fmask, bmask)
    
