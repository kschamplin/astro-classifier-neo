import numpy as np
import pandas as pd
import pytest
import torch

import pytorch_test.transformers as transformers

# some simple transformers for the container tests


@pytest.fixture
def add_1_transform():
    def fn(s):
        return s + 1
    return fn


@pytest.fixture
def multiply_transform():
    def fn(s):
        return s * 2
    return fn


@pytest.fixture
def random():
    return np.random.default_rng()


@pytest.fixture
def dataset():
    # this should return a dataset in pandas form
    pass


def test_sequential_transformer(add_1_transform, random):
    inputs = random.integers(5, size=(5))
    transformer = transformers.SequentialTransformer(
        [add_1_transform, add_1_transform])
    assert np.all(transformer(inputs) == inputs + 2)


def test_split_transformer(add_1_transform, multiply_transform, random):
    inputs = random.integers(5, size=(2, 3))
    transformer = transformers.SplitTransformer([
        add_1_transform,
        multiply_transform
    ])
    res = transformer(np.array(inputs))
    assert np.all(res[0] == add_1_transform(inputs[0]))
    assert np.all(res[1] == multiply_transform(inputs[1]))


def test_tensor_transformer(random):
    inputs = random.integers(3)
    transformer = transformers.TensorTransformer()
    res = transformer(inputs)
    assert torch.equal(res, torch.as_tensor(inputs))


def test_label_binarizer_transformer(random):
    inputs = random.integers(5)
    transformer = transformers.LabelBinarizerTransformer(range(5))
    res = np.array(transformer(inputs))
    assert res.shape == (5,)  # the shape is correct (5 classes)
    assert res[inputs] == 1  # the correct value is one (deterministic)
    assert np.sum(res) == 1  # only one index is one.
    assert np.sum(transformer([6])) == 0  # failed classes get empty arrays


def test_interpolate_transformer(random):
    mask = random.integers(2, size=(10))
    inputs = np.stack([
        np.arange(10),  # "index" column
        random.random(size=(10)) * mask  # random values with some masked.
    ])
    transformer = transformers.InterpolateTransformer(interp_cols=[1])
    res = transformer(inputs)
    assert res is not None


def test_pandas_split_transformer(random):
    data = {
        "foo": random.integers(5),
        "bar": random.integers(5)
    }
    transformer = transformers.PandasSplitTransformer([['foo'], ['bar']])
    res = transformer(pd.Series(data))
    assert res[0]['foo'] == data['foo']
    assert res[1]['bar'] == data['bar']
    # test multiple group (only works on pandas series)
    transformer = transformers.PandasSplitTransformer(
        [['foo', 'bar'], ['bar']])
    res = transformer(pd.Series(data))
    assert res[0]['foo'] == data['foo']
    assert res[0]['bar'] == data['bar']
    assert res[1]['bar'] == data['bar']
