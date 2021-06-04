from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_test import datasets, model

transform = datasets.get_plasticc_transformer()

data_path = Path("../data/plasticc_train_lightcurves.parquet")
train_dataset = datasets.PlasticcDataset(data_path, transform=transform)

train_dataloader = datasets.get_plasticc_dataloader(
    train_dataset, batch_size=100, num_workers=8)
logger = pl.loggers.TensorBoardLogger('tb_logs', 'mymodel')
m = model.DoubleLSTMNet(batch_size=100)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=50)

trainer.fit(m, train_dataloader)

print("Training complete")
