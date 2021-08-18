# TCN module
# based heavily on https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.nn.utils import weight_norm

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x[:, :, : - self.conv.padding[0]].contiguous()
        return x


class TemporalBlock(nn.Module):
    """Residual block for TCN"""

    def __init__(self, n_inputs, n_outputs, k_size, n_blocks=2, stride=1, dilation=1, dropout=0.2):
        super().__init__()
        # because we are using "causal convolutions" we need to pad a certain amount but only in one direction.

        net = []
        for i in range(n_blocks):
            conv = weight_norm(CausalConv1d(n_inputs, n_outputs, k_size, dilation, stride=stride))
            conv.weight.data.normal_(0, 0.01)
            relu = nn.ReLU()
            drop = nn.Dropout(dropout)
            net += [conv, relu, drop]

        self.net = nn.Sequential(*net)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        layers = []
