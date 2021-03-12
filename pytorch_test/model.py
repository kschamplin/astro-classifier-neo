# Model generators for training/testing.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleLSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        # primary lstm layer
        self.lstm = nn.LSTM(7, 200, 2) # 7 features, 200 hidden layers, 2 stacked.
        self.dense = nn.Linear(200,15)
    def forward(self, x, hidden):
        x, hidden = self.lstm(x.float(), hidden)
        # pad the packed sequence.
        x = nn.utils.rnn.pad_packed_sequence(x)[0]
        x = self.dense(x)
        return x, hidden
