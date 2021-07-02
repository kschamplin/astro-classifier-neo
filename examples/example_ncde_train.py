# train the NCDE model

from pathlib import Path

import pytorch_lightning as pl

from pytorch_test import plasticc, model
from pytorch_test.plasticc import PlasticcDataModule

plasticcDS = PlasticcDataModule(Path("./data"), num_workers=16, batch_size=200)

logger = pl.loggers.TensorBoardLogger('tb_logs', 'mymodel')

m = model.NCDE(input_channels=25, hidden_channels=50)
trainer = pl.Trainer(logger=logger, max_epochs=50)

trainer.fit(m, plasticcDS)

print("Training complete")
