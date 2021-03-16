# Dataset classes for plasticc/others

import torch
import pyarrow.parquet as pq
import numpy as np
from pytorch_test.transformers import *
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader

label_map = {
    90: "SNIa",
    67: "SNIa-91bg",
    52: "SNIax",
    42: "SNII",
    62: "SNIbc",
    95: "SLSN-I",
    15: "TDE",
    64: "KN",
    88: "AGN",
    # 92: "RRL",
    65: "M-dwarf",
    # 16: "EB",
    # 53: "Mira",
    6: "μLens-Single",
    991: "μLens-Binary",
    992: "ILOT",
    993: "CaRT",
    994: "PISN",
    # 995: "μLens-String"
}

def get_plasticc_transformer():
    """creates and returns a sane transformer for plasticc"""
    cols = [
        'true_target',
        'mjd',
        'flux',
        'passband',
        'detected_bool'
    ]
    x_y_splitter = pandas_split_transformer([cols[1:], [cols[0]]])
    x_transformer = sequential_transformer([  # x (input)
        pandas_numpy_transformer(),
        pivot_transformer(val_idx=1, col_idx=2, row_idx=0),
        interpolate_transformer(interp_cols=[1, 2, 3, 4, 5]),
        tensor_transformer()
    ])
    y_transformer = sequential_transformer([  # y (true values)
        pandas_numpy_transformer(),
        numpy_dtype_transformer(int),
        label_binarizer_transformer(list(label_map.keys())),
        tensor_transformer()
    ])

    return sequential_transformer([
        x_y_splitter,
        split_transformer([
            x_transformer,
            y_transformer
        ])
    ])

class plasticc_dataset(torch.utils.data.Dataset):
    def __init__(self, file, transform=None, cols=None):
        """Create a plasticc dataset from the given file.
        transforms argument specifies a transformer that will be called when data
        is accessed."""
        # load the file
        if cols is None:
            cols = [
                'true_target',
                'mjd',
                'flux',
                'passband',
                'detected_bool'
            ]
        self.transform = transform
        self.data = pq.read_table(file, columns=cols).to_pandas()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data.loc[idx])
        else:
            return self.data.loc[idx]

def get_plasticc_dataloader(dataset, batch_size=10):
    """Creates a dataloader that pads and batches the dataset."""
    def collate(batch):
        # batch contains a list of tuples of structure (sequence, target)
        data = [item[0] for item in batch]
        # pytorch has 'pack' and 'pad'. Pack is more optimal for LSTM as it reduces the # of ops
        data = pad_sequence(data)
        targets = torch.stack([item[1] for item in batch])
        return [data, targets]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)