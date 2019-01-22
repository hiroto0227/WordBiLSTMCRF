import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextualLSTM(nn.Module):
    def __init__(self, alphabet_size, pretrain_embedding, embedding_dim, hidden_dim, dropout, gpu):
        super(ContextualLSTM, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim // 2
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_embedding is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))
        #self.f_lstm = nn.GRU(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.f_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        #self.b_lstm = nn.GRU(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.b_lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        if self.gpu:
            self.drop = self.drop.cuda()
            self.embeddings = self.embeddings.cuda()
            self.lstm = self.lstm.cuda()

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
        batch_size = int(inputs.size(0))
        # forward
        embeds = self.drop(self.embeddings(inputs))
        f_hidden = None
        pack_input = pack_padded_sequence(embeds, seq_length, True)
        f_lstm_out, f_hidden = self.f_lstm(pack_input, f_hidden)
        f_lstm_out, _ = pad_packed_sequence(f_lstm_out)
        f_lstm_out = f_lstm_out.transpose(1, 0)
        f_masked_out = torch.zeros(batch_size, out_seq_length, self.hidden_dim)
        for batch_i, (out, mask) in enumerate(zip(f_lstm_out, fmask)):
            mask = mask.unsqueeze(1).expand(-1, self.hidden_dim).byte()
            masked = torch.masked_select(out, mask).view(-1, self.hidden_dim)
            f_masked_out[batch_i, :masked.shape[0]] = masked
        # backward
        b_hidden = None
        reverse_inputs = torch.zeros(batch_size, inputs.shape[1]).long()
        reverse_bmask = torch.zeros(batch_size, inputs.shape[1]).byte()
        for batch_i in range(batch_size):
            reverse_idx = torch.LongTensor([idx for idx in range(seq_length[batch_i]-1, -1, -1)])
            reverse_inputs[batch_i, :seq_length[batch_i]] = inputs[batch_i].index_select(0, reverse_idx)
            reverse_bmask[batch_i, :seq_length[batch_i]] = bmask[batch_i].index_select(0, reverse_idx)
        embeds = self.drop(self.embeddings(reverse_inputs))
        pack_input = pack_padded_sequence(embeds, seq_length, True)
        b_lstm_out, b_hidden = self.b_lstm(pack_input, b_hidden)
        b_lstm_out, _ = pad_packed_sequence(b_lstm_out)
        b_lstm_out = b_lstm_out.transpose(1, 0)
        b_masked_out = torch.zeros(batch_size, out_seq_length, self.hidden_dim)
        for batch_i, (out, mask) in enumerate(zip(b_lstm_out, fmask)):
            mask = mask.unsqueeze(1).expand(-1, self.hidden_dim).byte()
            masked = torch.masked_select(out, mask).view(-1, self.hidden_dim)
            reverse_idx = torch.LongTensor([idx for idx in range(masked.size(0)-1, -1, -1)])
            b_masked_out[batch_i, :masked.shape[0]] = masked.index_select(0, reverse_idx)

        out = torch.cat([f_masked_out, b_masked_out], dim=2)
        return out

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
    
