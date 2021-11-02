#!/usr/bin/env python3
# this file converts the data to spline coefficients for use in NCDE applications.

import torch
from torchcde import hermite_cubic_coefficients_with_backward_differences
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

from pytorch_test.plasticc import PlasticcTransformer, class_id_to_target
pt = PlasticcTransformer()


def generate_spline(curve, meta):
    """takes a curve and meta dataframe input and produces a new curve frame with spline coefficients"""
    res = pt(curve, meta)
    return hermite_cubic_coefficients_with_backward_differences(res[0]), res[1]

# iterate through all files, then for each curve in each file.


def convert_set(filepair):
    curve_file, meta_file = filepair
    curve_df = pd.read_feather(curve_file)
    curve_df.columns = curve_df.columns.map(eval)
    meta_df = pd.read_feather(meta_file)
    result = []
    n_parts = 0
    iterator = tqdm(meta_df.iterrows())
    for index, meta in iterator:
        if meta['target'] not in class_id_to_target.keys():
            continue
        curve = curve_df[curve_df[('object_id', '')] == meta['object_id']]

        result.append(generate_spline(curve, meta))
        if len(result) >= 1000:
            torch.save(result, f"{curve_file}_{n_parts}.pt")
            iterator.set_description(f"processed batch {n_parts}")
            n_parts = n_parts + 1
            result = []


    # result has a bunch of (coeff, target) tuples. we need to save them.
    # torch.save(result, f"{curve_file}.pt")


if __name__ == "__main__":
    # find all meta and curve files.
    sets = []
    for i in range(1, 12):
        s = f"data/plasticc_test_{i:02}"
        sets.append((s + "_curves.feather", s + "_meta.feather"))
    sets.append(("data/plasticc_train_curves.feather", "data/plasticc_train_meta.feather"))
    with Pool(4) as p:
        p.map(convert_set, sets)
    print("done")

