"""Datasets and loaders for various astronomical transient surveys.
At the moment it only supports PLAsTiCC."""
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_test.transformers as tf
# The way the data is structured is rather unfortunate - not continuous, and wide range. Thus, we will
# define two more ways of representing classes - one-hot and target index.
# one-hot is the output of the model, it looks like [0,0,0,1] where the index of 1 is the chosen class.
# target index in the prev. case is 4, since the idx max is 4.
# finally, we have the string name. we will define functions to convert all between.
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
    # 65: "M-dwarf",
    # 16: "EB",
    # 53: "Mira",
    6: "μLens-Single",
    991: "μLens-Binary",
    992: "ILOT",
    993: "CaRT",
    994: "PISN",
    # 995: "μLens-String"
}
class_weights = {
    6: 170.16862186163797,
    15: 16.35778047109659,
    42: 0.22169645981674177,
    52: 3.4828115463325315,
    62: 1.266346729674999,
    64: 1667.140708915145,
    67: 5.516625140838312,
    88: 2.1861661370653325,
    90: 0.13358571703126057,
    95: 6.196683088863515,
    991: 416.0032162958992,
    992: 130.27597784119524,
    993: 22.90596221959858,
    994: 189.18917601170162
}
# we define a dict of class_id: target mappings using a cheap hack.

label_targets = list(label_map.keys())
class_id_to_target = dict(zip(label_targets, range(len(label_targets))))

class_weights_target = {class_id_to_target[class_id]: weight for class_id, weight in class_weights.items()}
class_weights_target_list = torch.tensor([class_weights_target[x] for x in range(len(class_weights_target))])
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
        lambda x: class_id_to_target[x[0]],
        # tf.LabelBinarizerTransformer(list(label_map.keys())),
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


def get_plasticc_dataloader(dataset, batch_size=10, **kwargs):
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
        dataset, batch_size=batch_size, collate_fn=collate, **kwargs)
