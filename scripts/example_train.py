import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from pytorch_test import datasets
from pytorch_test import model
from pathlib import Path
from tqdm import tqdm
m = model.DoubleLSTMNet()

transform = datasets.get_plasticc_transformer()

data_path = Path("data/plasticc_train_lightcurves.parquet")
train_dataset = datasets.plasticc_dataset(data_path, transform=transform)

train_dataloader = datasets.get_plasticc_dataloader(train_dataset, batch_size=50)

optimizer = torch.optim.Adam(m.parameters(), lr=0.0001)

pbar = tqdm(train_dataloader)
for xb,yb in pbar:
    optimizer.zero_grad()
    # create initial states for the hidden layer based on input
    h0, c0 = m.init_hidden(bs=xb.shape[1])
    result, (h0,c0) = m(xb, (h0,c0))
    res = result[-1, :] # get the last timestep.
    loss = model.multiLogLoss(res, yb)
    pbar.write(str(loss))
    loss.backward()
    optimizer.step()

print("Training complete")