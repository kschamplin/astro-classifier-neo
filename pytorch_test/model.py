# Model generators for training/testing.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleLSTMNet(nn.Module):
    def __init__(self, input_size=7, hidden_size=200, num_layers=2,
                 num_classes=15, batch_size=20, device=None):
        super().__init__()
        # primary lstm layer)
        # 7 features, 200 hidden layers, 2 stacked.
        if device is None:
            device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.dense = nn.Linear(hidden_size, num_classes).to(device)
        self.batch_size = batch_size
        self.device = device

    def init_hidden(self, bs=10):
        h0 = torch.zeros(self.lstm.num_layers, bs, self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, bs, self.lstm.hidden_size)
        return (h0, c0)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x.float(), hidden)
        # pad the packed sequence.
        # x = nn.utils.rnn.pad_packed_sequence(x)[0]
        x = self.dense(x)
        x = F.softmax(x, dim=2)
        return x, hidden


def multiLogLoss(pred, target):
    return (-(pred + 1e-24).log() * target).sum(dim=1).mean()
