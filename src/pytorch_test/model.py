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

# NCDE Stuff


class NCDEFunction(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(F, self).__init__()
        # For illustrative purposes only. You should usually use an MLP or something. A single linear layer won't be
        # that great.
        self.linear = torch.nn.Linear(hidden_channels,
                                          hidden_channels * input_channels)
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

    def forward(self, t, z):
        batch_dims = z.shape[:-1]
        return self.linear(z).tanh().view(*batch_dims, self.hidden_channels, self.input_channels)


class NCDE(pl.LightningModule):
    """Neural Controlled Differential Equation model for classification on irregular, multi-modal time series"""

    def __init__(self, input_channels=7, hidden_channels=128, output_channels=14, interpolation="cubic"):
        super().__init__()

        self.initial = nn.Linear(input_channels, hidden_channels)
        self.model = NCDEFunction(input_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, output_channels)

        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_target_list))

        self.interpolation = interpolation

    def forward(self, x):
        # NOTE: x should be the natural cubic spline coefficients. Look into datasets.py for how to generate these.
        x = torchcde.natural_cubic_coeffs(x)
        if self.interpolation == "cubic":
            x = torchcde.NaturalCubicSpline(x)
        elif self.interpolation == "linear":
            x = torchcde.LinearInterpolation(x)
        else:
            raise ValueError("invalid interpolation given")

        x0 = x.evaluate(x.interval[0])
        z0 = self.initial(x0)
        zt = torchcde.cdeint(X=x, func=self.model, z0=z0, t=x.interval)

        return self.output(zt[..., -1, :])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x).squeeze(-1)
        loss = self.loss(pred_y, y)
        return loss

# Variational Autoencoder stuff.


class VEncoder(torch.nn.Module):
    """GRU Encoder for VAE"""
    def __init__(self, input_channels, hidden_size, latent_dims):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_size, batch_first=True)

        self.z_mean = nn.Linear(hidden_size, latent_dims)
        self.z_std = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x, _ = self.gru(x)
        x = F.relu(x[-1])
        return self.z_mean(x), self.z_std(x)


class VDecoder(torch.nn.Module):
    """GRU Decoder for VAE"""
    def __init__(self, latent_dims, hidden_size, output_length):
        super().__init__()
        self.gru = nn.GRU(latent_dims, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x


class AutoEncoder(pl.LightningModule):
    """Variational Auto-encoder model used to augment dataset and fill in space."""

    def __init__(self, input_channels=7, latent_dims=50):
        super().__init__()
        self.encoder = VEncoder(input_channels, 150, latent_dims)
        self.decoder = VDecoder(latent_dims, 150, 200)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def kl_divergence(self, z, mu, std):
        "MC KL Divergence"

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = log_qzx - log_pz

        return kl.sum(-1)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        log_pzx = dist.log_prob(x)

        return log_pzx.sum(dim=(1,2,3))

    def training_step(self, batch, batch_idx):
        x, _ = batch  # discard true class (don't care)

        x_mean, x_std = self.encoder(x)  # get probs in latent space for x

        x_std = torch.exp(x_std / 2)
        q = torch.distributions.Normal(x_mean, x_std)
        z = q.rsample()

        x_hat = self.decoder(z)

        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        kl_loss = self.kl_divergence(z, x_mean, x_std)

        elbo = kl_loss - recon_loss
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl_loss,
            'recon': recon_loss
        })

        return elbo


