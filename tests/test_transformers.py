import pytest
import pytorch_test.transformers as transformers
import numpy as np
import torch
# some simple transformers for the container tests
@pytest.fixture
def add_1_transform():
    def fn(s):
        return s+1
    return fn
@pytest.fixture
def multiply_transform():
    def fn(s):
        return s * 2
    return fn

@pytest.fixture
def random():
    return np.random.default_rng()

def test_sequential_transformer(add_1_transform, random):
    inputs = random.integers(5,size=(5))
    transformer = transformers.sequential_transformer([add_1_transform])
    assert np.all(transformer(inputs) == inputs + 1)

def test_split_transformer(add_1_transform, multiply_transform, random):
    inputs = random.integers(5, size=(2,3))
    transformer = transformers.split_transformer([
        add_1_transform,
        multiply_transform
    ])
    res = transformer(np.array(inputs))
    assert np.all(res[0] == add_1_transform(inputs[0]))
    assert np.all(res[1] == multiply_transform(inputs[1]))

def test_tensor_transformer(random):
    inputs = random.integers(3)
    transformer = transformers.tensor_transformer()
    res = transformer(inputs)
    assert torch.equal(res,torch.as_tensor(inputs))

def test_label_binarizer_transformer(random):
    inputs = random.integers(5)
    transformer = transformers.label_binarizer_transformer(range(5))
    res = np.array(transformer(inputs))
    assert res.shape == (1,5) # the shape is correct (5 classes)
    assert res[0,inputs] == 1 # the correct value is one (deterministic)
    assert np.sum(transformer([6])) == 0 # failed classes get empty arrays

def test_interpolate_transformer(random):
    mask = random.integers(2,size=(10))
    inputs = np.stack([
        np.arange(10), # "index" column
        random.random(size=(10)) * mask #random values with some masked.
        ])
    transformer = transformers.interpolate_transformer(interp_cols=[1])
    res = transformer(inputs)
