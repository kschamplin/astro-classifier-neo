import pytest
import pytorch_test.datasets as datasets
import pytorch_test.transformers as transformers
import pyarrow.parquet as pq
import pandas as pd
import torch


dataset_path = "plasticc_train_lightcurves.parquet"


@pytest.fixture
def example_dataset():
    return datasets.plasticc_dataset(dataset_path)


@pytest.fixture
def example_transformer():
    return transformers.split_transformer((
        transformers.sequential_transformer([  # x (input)
            transformers.pivot_transformer(val_idx=1, col_idx=2, row_idx=0),
            transformers.interpolate_transformer(interp_cols=[1, 2, 3, 4, 5]),
            transformers.tensor_transformer()
        ]),
        transformers.sequential_transformer([  # y (true values)
            transformers.label_binarizer_transformer(list(label_map.keys())),
            transformers.tensor_transformer()
        ])

    ))


@pytest.fixture
def example_transform_dataset(example_transformer):
    return datasets.plasticc_dataset(dataset_path, transform=example_transformer)


def test_dataset_creation(example_dataset):
    example_dataset


def test_dataset_length(example_dataset):
    real_length = pq.read_metadata(dataset_path).num_rows
    assert real_length == len(example_dataset)


def test_dataset_access(example_dataset):
    assert example_dataset[0].equals(example_dataset.data.loc[0])


def test_dataset_cols():
    ds = datasets.plasticc_dataset(dataset_path, cols=['true_target'])

    # invariant: accessing the first + checking the length guarentees accuracy.
    assert ds[0]['true_target'] == 88
    assert len(ds[0].index) == 1

def test_dataset_transform(example_transformer):
    ds = datasets.plasticc_dataset(dataset_path, transformers=example_transformer)
    # TODO: finish this
