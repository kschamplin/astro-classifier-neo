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

def test_sequential_transformer(add_1_transform):
    inputarray = np.array([1,2,3,4])
    transformer = transformers.sequential_transformer([add_1_transform])
    assert np.all(transformer(inputarray) == inputarray + 1)

def test_split_transformer(add_1_transform, multiply_transform):
    inputs = np.array([[1,2,3], [4,5,6]])
    transformer = transformers.split_transformer([
        add_1_transform,
        multiply_transform
    ])
    res = transformer(np.array(inputs))
    assert np.all(res[0] == add_1_transform(inputs[0]))
    assert np.all(res[1] == multiply_transform(inputs[1]))

def test_tensor_transformer():
    inputs = np.random.rand(3)
    transformer = transformers.tensor_transformer()
    res = transformer(inputs)
    assert torch.equal(res,torch.as_tensor(inputs))

def test_label_binarizer_transformer():
    inputs = np.random.randint(5)
    transformer = transformers.label_binarizer_transformer(range(5))
    res = np.array(transformer(inputs))
    assert res.shape == (1,5) # the shape is correct (5 classes)
    assert res[0,inputs] == 1 # the correct value is one (deterministic)

def test_interpolate_transformer():
    inputs = np.random.randint()
