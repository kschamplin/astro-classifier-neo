# Dataset classes for plasticc/others

import torch
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

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

class plasticc_dataset(torch.utils.data.Dataset):
    def __init__(self, file, transform=None):
        """Create a plasticc dataset from the given file.
        transforms argument specifies a transformer that will be called when data
        is accessed."""
        # load the file
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
        if self.transform is not None:
            return self.transform([
                self.data.loc[idx].drop('true_target').values,
                self.data.loc[idx, 'true_target']
            ])
        return (
            torch.from_numpy(self.data.loc[idx].drop('true_target').values),
            torch.tensor(self.data.loc[idx]['true_target'])
        )
