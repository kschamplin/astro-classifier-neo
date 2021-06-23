# Handles the plasticc dataset loading/conversion
import torch
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import pyarrow.feather as feather
from pytorch_lightning import LightningDataModule
from pathlib import Path
from tqdm.auto import tqdm
import torchcde
import hashlib
import requests
import shutil
import gzip
import re
# Constants needed for proper mappings and labels.

from .constants import passband_map, class_id_to_target, label_targets


class PlasticcDataModule(LightningDataModule):
    def __init__(self, data_path: Path = Path("./data"), download=False, batch_size: int = 50, num_workers=8):
        super().__init__()
        self.data_path = Path(data_path)
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download the CSV if the feather files arent there yet.
        # process and save
        if self.download:
            # _download(self.data_path)
            # _gunzip(self.data_path)
            _zenodo_convert(self.data_path)

    def setup(self, stage=None):
        transform = _transform
        # construct the testing dataset by using concat dataset.
        self.plasticc_train = PlasticcDataset("plasticc_train", self.data_path, transform)
        plasticc_test = data.ConcatDataset(
            [PlasticcDataset(f"plasticc_test_{i:02}", self.data_path, transform) for i in range(1, 2)])
        l = len(plasticc_test)
        self.plasticc_test, self.plasticc_val = data.random_split(plasticc_test, [int(0.9 * l), int(0.1 * l) + 1])

    def train_dataloader(self):
        return data.DataLoader(self.plasticc_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.plasticc_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.plasticc_test, batch_size=self.batch_size, num_workers=self.num_workers)


# next we have all the helper functions.
def _transform(curve, meta):
    "Converts the dataframe/series into tensors"
    curve = torch.tensor(curve[['mjd', 'flux']].values)
    curve[:, 0] = curve[:, 0] - curve[0, 0]  # the time always starts at zero and goes up.

    # generate observation masks. the current shape is (len,channels) where the first channel is time. So we must
    # take the remaining channels and generate an observation mask (!= nan)
    for channel in range(1, curve.shape[1]):
        mask = (~curve[:, channel].isnan()).cumsum(0)
        curve = torch.cat((curve, mask.unsqueeze(-1)), 1)

    # fill forward to fixed length (500)
    curve = torch.cat([curve, curve[-1].unsqueeze(0).expand(500 - curve.size(0), curve.size(1))])
    # fillnan (TODO Remove for ncde)
    curve = torch.nan_to_num(curve)
    m = int(meta['true_target'])
    # convert the target to the 0-14 class layout.
    m = torch.tensor(class_id_to_target[m])
    return curve, m


blacklist = [
    "plasticc_modelpar.tar.gz"
]

def _download(data_dir):
    # create data dir
    if not data_dir.exists():
        data_dir.mkdir()
    # scrape the file list
    print("Gathering file list")
    files = requests.get(
        "https://zenodo.org/api/records/2539456").json()['files']
    files = [f for f in files if f['key'] not in blacklist]

    for f in tqdm(files):
        # TODO: add checksum validation
        r = requests.get(f['links']['self'], stream=True)
        size = int(r.headers.get('content-length', 0))
        bs = 1024
        pbar = tqdm(total=size, unit="iB", unit_scale=True)
        pbar.set_description(f"{f['key']}")
        with (data_dir / f['key']).open(mode='wb') as dest:
            md5 = hashlib.md5()
            for data in r.iter_content(bs):
                dest.write(data)
                pbar.update(len(data))
                md5.update(data)
            print(md5.hexdigest())
        pbar.close()


def _gunzip(data_dir):
    # unzips all the files
    # get list of all gz files
    gzs = data_dir.glob('*.gz')
    for gzpath in tqdm(gzs):
        with gzip.open(gzpath, 'rb') as input_file:
            with open(gzpath.parent / gzpath.stem, 'wb') as output:
                shutil.copyfileobj(input_file, output)


def _process_plasticc_csv(curve_table: pd.DataFrame, meta_table: pd.DataFrame, name: str, data_dir: Path):
    "Converts a plasticc curve/metadata table to pivoted feather tables"
    curve_table['passband'] = curve_table['passband'].map(passband_map)
    curve_ids = curve_table['object_id'].unique()

    pivoted_result = curve_table.groupby(['object_id', 'mjd', 'passband']).sum().unstack('passband').reset_index()
    new_meta = meta_table[meta_table['object_id'].isin(curve_ids)]  # only metadata for curves that we have.

    feather.write_feather(pivoted_result, data_dir / f"{name}_curves.feather", compression="zstd")
    feather.write_feather(new_meta, data_dir / f"{name}_meta.feather", compression="zstd")


def _zenodo_convert(data_dir: Path):
    """Converts the original plasticc CSVs to our new format"""
    train_curves = pd.read_csv(data_dir / "plasticc_train_lightcurves.csv")
    train_meta = pd.read_csv(data_dir / "plasticc_train_metadata.csv")
    _process_plasticc_csv(train_curves, train_meta, "plasticc_train", data_dir=data_dir)
    test_curves = [(x, re.search("(\d+)", str(x))) for x in data_dir.glob("plasticc_test_light*.csv")]
    test_meta = pd.read_csv(data_dir / "plasticc_test_metadata.csv")

    for cur in test_curves:
        df = pd.read_csv(cur[0])
        _process_plasticc_csv(df, test_meta, f"plasticc_test_{cur[1].group(0)}", data_dir=data_dir)


class PlasticcDataset(torch.utils.data.Dataset):
    "Loads a single curve/meta pair from the converted plasticc dataset"

    def __init__(self, file_name, data_dir, transform=None):
        # load the data using memory-map
        data_dir = Path(data_dir)
        self.curves = feather.read_feather(data_dir / f"{file_name}_curves.feather", memory_map=True)
        self.curves.columns = self.curves.columns.map(eval)  # hack to restore multiindex columns

        self.meta = feather.read_feather(data_dir / f"{file_name}_meta.feather", memory_map=True)

        # remove unwanted transients from the dataset
        self.meta = self.meta.loc[self.meta['true_target'].astype('int8').isin(label_targets)]
        self.curves = self.curves.loc[self.curves['object_id'].isin(self.meta['object_id'])]
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        res_meta = self.meta.iloc[idx]
        obj_id = res_meta['object_id']

        res_curve = self.curves[self.curves['object_id'] == obj_id]

        if self.transform is not None:
            return self.transform(res_curve, res_meta)
        return res_curve, res_meta

# phase 2: convert spline coefficients
# somehow get the results from above
# convert to numpy/torch, only taking flux.
# create masks for each observation channel
# subtract starting mjd from each time-series index (so that it starts from zero and counts days since then)
# generate coefficients
# save to new table {name}_curves_spline.feather
