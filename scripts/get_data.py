#!/usr/bin/env python3
# downloads plasticc data and optionally preprocesses it.
import gzip
import hashlib
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

blacklist = [
    "plasticc_modelpar.tar.gz"
]
data_dir = Path("./data")


def download():
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


def gunzip():
    # unzips all the files
    # get list of all gz files
    gzs = data_dir.glob('*.gz')
    for gzpath in tqdm(gzs):
        with gzip.open(gzpath, 'rb') as input_file:
            with open(gzpath.parent / gzpath.stem, 'wb') as output:
                shutil.copyfileobj(input_file, output)


if __name__ == "__main__":
    gunzip()
