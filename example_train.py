from pathlib import Path

import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_test import datasets, model
import pytorch_lightning as pl


transform = datasets.get_plasticc_transformer()

data_path = Path("plasticc_train_lightcurves.parquet")
train_dataset = datasets.PlasticcDataset(data_path, transform=transform)

train_dataloader = datasets.get_plasticc_dataloader(
    train_dataset, batch_size=20)


m = model.DoubleLSTMNet()
trainer = pl.Trainer(gpus=1, profiler="simple")

trainer.fit(m, train_dataloader)

print("Training complete")
