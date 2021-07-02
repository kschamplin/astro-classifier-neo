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
    at every time-step"""

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
        """Creates the initial lstm state given a batch size."""
        hidden_0 = torch.zeros(
            self.lstm.num_layers,
            bs,
            self.lstm.hidden_size).to(self.device)
        cell_0 = torch.zeros(
            self.lstm.num_layers,
            bs,
            self.lstm.hidden_size).to(self.device)
        return hidden_0, cell_0

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
        x, y = batch
        h0, c0 = self.init_hidden(x.shape[0])
        y_hat, _ = self(x, (h0, c0))
        y_hat = y_hat[:, -1]  # we want the last output for each one.
        # loss = multi_log_loss(y_hat, y)
        loss = self.loss(y_hat, y)
        self.logger.experiment.add_scalar("loss", loss, self.global_step)
        return loss


class GRUNet(pl.LightningModule):
    """A simple chained GRU network that tries to classify curves
    at every timestep"""

    def __init__(self, input_size=7, hidden_size=200, num_layers=2,
                 num_classes=14, *, batch_size=20, device=None):
        super().__init__()
        # primary lstm layer)
        # 7 features, 200 hidden layers, 2 stacked.
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, num_classes)
        self.batch_size = batch_size
        self.loss = nn.CrossEntropyLoss()

    def init_hidden(self, bs):
        """Creates the initial lstm state given a batch size."""
        hidden_0 = torch.zeros(
            self.gru.num_layers,
            bs,
            self.gru.hidden_size).to(self.device)
        return hidden_0

    def forward(self, tensor, lstm_state):
        """Compute forward-prop of input tensor."""
        tensor, lstm_state = self.gru(tensor.float(), lstm_state)
        # pad the packed sequence.
        # x = nn.utils.rnn.pad_packed_sequence(x)[0]
        tensor = self.dense(tensor)
        tensor = F.softmax(tensor, dim=2)
        return tensor, lstm_state

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        h0 = self.init_hidden(x.shape[0])
        y_hat, _ = self(x, h0)
        y_hat = y_hat[:, -1]  # we want the last output for each one.
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
        x = torchcde.NaturalCubicSpline(x)
        x0 = x.evaluate(x.interval[0])
        z0 = self.initial(x0)
        zt = torchcde.cdeint(X=x, func=self.func, z0=z0, t=x.interval)

        return self.output(zt[..., -1, :])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x).squeeze(-1)
        loss = self.loss(pred_y, y)
        return loss


class VEncoder(torch.Module):
    """GRU Encoder for VAE"""
    def __init__(self, input_channels, hidden_size, latent_dims):
        super(torch.Module, self).__init__()
        self.gru = nn.GRU(input_channels, hidden_size, batch_first=True)

        self.z_mean = nn.Linear(hidden_size, latent_dims)
        self.z_var = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x = F.relu(self.gru(x))
        return self.z_mean(x), self.z_var(x)


class VDecoder(torch.Module):
    """GRU Decoder for VAE"""
    def __init__(self, latent_dims, hidden_size, output_length):
        self.gru = nn.GRU(latent_dims, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        x = self.gru(x)
        x = self.linear(x)
        return x


class AutoEncoder(pl.LightningModule):
    """Variational Auto-encoder model used to augment dataset and fill in space."""

    def __init__(self, input_channels=7, latent_dims=50):
        super().__init__()
        self.encoder = VEncoder(input_channels, 150, latent_dims)
        self.decoder = VDecoder(latent_dims, 150, 200)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


