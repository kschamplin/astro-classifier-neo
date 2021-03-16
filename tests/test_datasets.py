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
    return datasets.get_plasticc_transformer()


@pytest.fixture
def example_transform_dataset(example_transformer):
    return datasets.plasticc_dataset(
        dataset_path, transform=example_transformer)


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
    ds = datasets.plasticc_dataset(dataset_path, transform=example_transformer)
    # the output is one-hot.
    assert len(ds[0][1]) == len(datasets.label_map.keys())
    assert ds[0][0].shape
    # TODO: finish this


def test_dataloader():
    assert True
