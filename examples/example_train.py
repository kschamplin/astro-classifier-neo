# train a model on the plasticc dataset (assume it is downloaded already)

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_test import plasticc, model
from pytorch_test.plasticc import PlasticcDataModule

plasticcDS = PlasticcDataModule("./data", num_workers=16, batch_size=200)

logger = pl.loggers.TensorBoardLogger('tb_logs', 'mymodel')
m = model.DoubleLSTMNet(batch_size=200, input_size=13, hidden_size=500)
trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=50)

trainer.fit(m, plasticcDS)

print("Training complete")
