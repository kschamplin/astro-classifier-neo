"""Datasets and loaders for various astronomical transient surveys.
At the moment it only supports PLAsTiCC."""
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_test.transformers as tf

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
    # split true target from the rest.
    x_y_splitter = tf.PandasSplitTransformer([cols[1:], [cols[0]]])
    x_transformer = tf.SequentialTransformer([  # x (input)
        tf.PandasNumpyTransformer(),
        tf.PivotTransformer(val_idx=1, col_idx=2, row_idx=0),
        tf.DiffTransformer(0),
        tf.InterpolateTransformer(interp_cols=[1, 2, 3, 4, 5]),
        tf.TensorTransformer()
    ])
    y_transformer = tf.SequentialTransformer([  # y (true values)
        tf.PandasNumpyTransformer(),
        tf.NumpyDtypeTransformer(int),
        tf.LabelBinarizerTransformer(list(label_map.keys())),
        tf.TensorTransformer()
    ])

    return tf.SequentialTransformer([
        x_y_splitter,
        tf.SplitTransformer([
            x_transformer,
            y_transformer
        ])
    ])


class PlasticcDataset(torch.utils.data.Dataset):
    """Class representing a plasticc dataset. Expected to be
    from a parquet file."""

    def __init__(self, file, transform=None, cols=None):
        """Create a plasticc dataset from the given file.
        transforms argument specifies a transformer that will be
        called when data is accessed."""
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
        return self.data.loc[idx]


def get_plasticc_dataloader(dataset, batch_size=10):
    """Creates a dataloader that pads and batches the dataset."""
    def collate(batch):
        # batch contains a list of tuples of structure (sequence, target)
        data = [item[0] for item in batch]
        # pytorch has 'pack' and 'pad'. Pack is more optimal for LSTM as it
        # reduces the # of ops
        data = pad_sequence(data)
        targets = torch.stack([item[1] for item in batch])
        return [data, targets]
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate)
