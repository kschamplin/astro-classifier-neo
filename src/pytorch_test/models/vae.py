import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F


class VEncoder(torch.nn.Module):
    """GRU Encoder for VAE"""
    def __init__(self, input_channels, hidden_size, latent_dims):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_size, batch_first=True)

        self.z_mean = nn.Linear(hidden_size, latent_dims)
        self.z_std = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x, _ = self.gru(x)
        x = F.relu(x[:, -1])
        return self.z_mean(x), self.z_std(x)


class VDecoder(torch.nn.Module):
    """GRU Decoder for VAE"""
    def __init__(self, latent_dims, hidden_size, output_length, sequence_length=500):
        super().__init__()

        self.gru = nn.GRU(latent_dims, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_length)
        self.sequence_length = sequence_length

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
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

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.log_scale)
        dist = torch.distributions.Normal(x_hat, scale)

        print(dist.batch_shape, dist.event_shape, x.shape, x_hat.shape)
        log_pzx = dist.log_prob(x)

        return log_pzx.sum(dim=(1,2,3))

    def training_step(self, batch, batch_idx):
        x, _ = batch  # discard true class (don't care)

        x_mean, x_std = self.encoder(x)  # get probs in latent space for x

        x_std = torch.exp(x_std / 2) # log std dev
        q = torch.distributions.Normal(x_mean, x_std)
        z = q.rsample() # sample the distribution

        x_hat = self.decoder(z)

        recon_loss = self.gaussian_likelihood(x_hat, x)
        kl_loss = self.kl_divergence(z, x_mean, x_std)

        elbo = kl_loss - recon_loss
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl_loss,
            'recon': recon_loss
        })

        return elbo