import torch
import torch.nn as nn
import torch.nn.functional as F

class AtNet(nn.Module):
    def __init__(self, dict_weight, dropout=0.2, lstm_units=1024, dense_units=30):
        super(AtNet, self).__init__()
        dict_shape = dict_weight.shape
        self.dropout = dropout
        self.hidden_dim = lstm_units
        self.emb = nn.Embedding(dict_shape[0], dict_shape[1])

        if dict_weight is not None:
            word_embedding = torch.FloatTensor(dict_weight)
            self.emb.weight.data = word_embedding
            self.emb.weight.requires_grad = True

        self.lstm = nn.LSTM(dict_shape[1], lstm_units, 1, batch_first=True)
        self.dense = nn.Linear(lstm_units, dense_units)
        self.out = nn.Linear(dense_units, 2)

    def forward(self, batch, perturbation=None):
        embedding = self.emb(batch)  # [batch, len, 256]
        drop = F.dropout(embedding, p=self.dropout)
        if perturbation is not None:
            drop += perturbation
        lstm, _ = self.lstm(drop)
        max_lstm = F.max_pool1d(lstm.transpose(1, 2), kernel_size=embedding.size(1)).squeeze()
        dense = F.relu(self.dense(max_lstm))  # [batch, 30]
        # dense bs*L*dim     ==>   bs*dim*L  ==> bs*dim*1  squeeze
        pred = F.sigmoid(self.out(dense)).squeeze()  # [batch, 2]
        return pred, embedding

