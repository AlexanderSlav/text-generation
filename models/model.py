import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                 drop_prob=0.5, use_gru=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.dropout = nn.Dropout(drop_prob)
        self.rnn = nn.LSTM(len(self.chars), n_hidden, n_layers,
                           dropout=drop_prob, batch_first=True) if not use_gru else nn.GRU(len(self.chars), n_hidden,
                                                                                           n_layers,
                                                                                           dropout=drop_prob,
                                                                                           batch_first=True)
        self.fc = nn.Linear(n_hidden, len(self.chars))

        # self.init_weights()

    def forward(self, x, hc):
        ''' Forward pass through the network '''

        x, (h, c) = self.rnn(x, hc)
        x = self.dropout(x)

        # Stack up LSTM outputs
        x = x.reshape(-1, self.n_hidden)
        # x = x.view(x.size()[0] * x.size()[1], self.n_hidden)

        x = self.fc(x)

        return x, (h, c)

    def init_hidden(self, sequence_length):
        return (torch.zeros(self.n_layers, sequence_length, self.n_hidden).to(device='cuda'),
                torch.zeros(self.n_layers, sequence_length, self.n_hidden).to(device='cuda'))
