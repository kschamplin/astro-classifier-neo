"""Contains sample models for comparison and benchmarking,
as well as useful loss/optimization utilities"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleLSTMNet(nn.Module):
    """A simple chained LSTM network that tries to classify curves
    at every timestep"""

    def __init__(self, input_size=7, hidden_size=200, num_layers=2,
                 num_classes=15, *, batch_size=20, device=None):
        super().__init__()
        # primary lstm layer)
        # 7 features, 200 hidden layers, 2 stacked.
        if device is None:
            device = torch.device('cpu')
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.dense = nn.Linear(hidden_size, num_classes).to(device)
        self.batch_size = batch_size
        self.device = device

    def init_hidden(self, batch_size=10):
        """Creates the initial lstm state given a batch size. Can be overwritten
        for more performance."""
        hidden_0 = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size)
        cell_0 = torch.zeros(
            self.lstm.num_layers,
            batch_size,
            self.lstm.hidden_size)
        return (hidden_0, cell_0)

    def forward(self, tensor, lstm_state):
        """Compute forward-prop of input tensor."""
        tensor, lstm_state = self.lstm(tensor.float(), lstm_state)
        # pad the packed sequence.
        # x = nn.utils.rnn.pad_packed_sequence(x)[0]
        tensor = self.dense(tensor)
        tensor = F.softmax(tensor, dim=2)
        return tensor, lstm_state


def multi_log_loss(pred, target):
    """Computes the multi-class log loss from two one-hot vectors."""
    return (-(pred + 1e-24).log() * target).sum(dim=1).mean()
