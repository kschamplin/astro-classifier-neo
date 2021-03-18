from pathlib import Path

import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_test import datasets, model

w = SummaryWriter()

m = model.DoubleLSTMNet()


transform = datasets.get_plasticc_transformer()

data_path = Path("data/plasticc_train_lightcurves.parquet")
train_dataset = datasets.PlasticcDataset(data_path, transform=transform)

train_dataloader = datasets.get_plasticc_dataloader(
    train_dataset, batch_size=50)


optimizer = torch.optim.Adam(m.parameters(), lr=0.0005)

pbar = tqdm(train_dataloader)
i = 0
for idx, (xb, yb) in enumerate(pbar):
    optimizer.zero_grad()
    # create initial states for the hidden layer based on input
    h0, c0 = m.init_hidden(batch_size=xb.shape[1])
    result, (h0, c0) = m(xb, (h0, c0))
    res = result[-1, :]  # get the last timestep.
    loss = model.multi_log_loss(res, yb)
    pbar.write(str(loss))
    w.add_scalar("loss", loss, i)
    i = i + 1
    loss.backward()
    optimizer.step()

print("Training complete")
w.close()
