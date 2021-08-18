import pytorch_lightning as pl
import torch
import torchcde
from torch import nn as nn
from torch.nn import functional as F

from pytorch_test.plasticc.constants import class_weights_target_list


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