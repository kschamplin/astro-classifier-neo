# VAE training example

from pathlib import Path

import pytorch_lightning as pl

from pytorch_test.model import AutoEncoder
from pytorch_test.plasticc import PlasticcDataModule

plasticcDS = PlasticcDataModule("./data", num_workers=4, batch_size=200)

logger = pl.loggers.TensorBoardLogger('tb_logs', 'vae')

m = AutoEncoder(input_channels=25, latent_dims=50)

trainer = pl.Trainer(gpus=0, logger=logger, max_epochs=50)

trainer.fit(m, plasticcDS)

print("Training complete")