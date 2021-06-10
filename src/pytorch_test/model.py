"""Contains sample models for comparison and benchmarking,
as well as useful loss/optimization utilities"""
import pytorch_lightning as pl
import torch
import torchcde
import torch.nn as nn
import torch.nn.functional as F
from pytorch_test.plasticc.constants import class_weights_target_list

class DoubleLSTMNet(pl.LightningModule):
    """A simple chained LSTM network that tries to classify curves
    at every timestep"""

    def __init__(self, input_size=7, hidden_size=200, num_layers=2,
                 num_classes=14, *, batch_size=20, device=None):
        super().__init__()
        # primary lstm layer)
        # 7 features, 200 hidden layers, 2 stacked.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, num_classes)
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_target_list))

    def init_hidden(self, bs):
        """Creates the initial lstm state given a batch size. Can be overwritten
        for more performance."""
        hidden_0 = torch.zeros(
            self.lstm.num_layers,
            bs,
            self.lstm.hidden_size).to(self.device)
        cell_0 = torch.zeros(
            self.lstm.num_layers,
            bs,
            self.lstm.hidden_size).to(self.device)
        return (hidden_0, cell_0)

    def forward(self, tensor, lstm_state):
        """Compute forward-prop of input tensor."""
        tensor, lstm_state = self.lstm(tensor.float(), lstm_state)
        # pad the packed sequence.
        # x = nn.utils.rnn.pad_packed_sequence(x)[0]
        tensor = self.dense(tensor)
        tensor = F.softmax(tensor, dim=2)
        return tensor, lstm_state

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):

        x,y = batch
        h0, c0 = self.init_hidden(x.shape[0])
        y_hat, _ = self(x, (h0,c0))
        y_hat = y_hat[:,-1] # we want the last output for each one.
        # loss = multi_log_loss(y_hat, y)
        loss = self.loss(y_hat, y)
        self.logger.experiment.add_scalar("loss", loss, self.global_step)
        return loss

def multi_log_loss(pred, target):
    """Computes the multi-class log loss from two one-hot vectors."""
    return (-(pred + 1e-24).log() * target).sum(dim=1).mean()


class NCDE(pl.LightningModule):
    """Neural Controlled Differential Equation model for classification on irregular, multi-modal time series"""

    def __init__(self, input_channels=7, hidden_channels=128, output_channels=14):
        super().__init__()

        self.initial = nn.Linear(input_channels, hidden_channels)
        self.model = nn.Linear(hidden_channels, hidden_channels * input_channels)
        self.output = nn.Linear(hidden_channels, output_channels)

        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_target_list))

    def forward(self, x):

        # NOTE: x should be the natural cubic spline coefficients. Look into datasets.py for how to generate these.
        X = torchcde.NaturalCubicSpline(x)
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
        
        return self.output(zt[...,-1,:])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x).squeeze(-1)
        loss = self.loss(pred_y, y)
        return loss