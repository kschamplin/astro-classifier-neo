# Basic data transformers for constructing an input pipeline.
# Copyright (C) Saji Champlin. All rights reserved.
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import torch
import pandas as pd
from typing import Any
from collections.abc import Callable

class sequential_transformer(object):
    """Applies a series of transformations on input data."""

    def __init__(self, functions: list[Callable[Any, Any]]) -> None:
        # everything in here should be callable
        assert all(callable(x) for x in functions)
        self.functions = functions

    def __call__(self, x: Any) -> Any:
        for fn in self.functions:
            x = fn(x)
        return x


class split_transformer(object):
    """A transformer that splits the input (which should be some map-type object like a list)
    among its stored transformers. Useful for applying different transformations to different parts
    of the data."""

    def __init__(self, functions: list[Callable[Any, Any]]) -> None:
        # everything in here should be callable
        assert all(callable(x) for x in functions)
        self.functions = functions

    def __call__(self, x: list[Any]) -> list[Any]:
        assert len(x) == len(self.functions)
        for i in range(len(x)):
            x[i] = self.functions[i](x[i])
        return x


class pivot_transformer(object):
    """Takes the input (lists of arrays) and returns a pivot table."""

    def __init__(self, val_idx: int =0, col_idx:int=3,
                 row_idx:int=2, add_index_col:bool=True) -> None:
        """Creates a pivot transformer.
        parameters are the indexes that will be used to construct pivot table.
        add_index_col specifies whether there should be an extra column for index.
        Expects things to be shaped like (rows,cols)."""
        self.val_idx = val_idx
        self.col_idx = col_idx
        self.row_idx = row_idx
        self.add_index_col = add_index_col

    def __call__(self, s: list[np.ndarray]) -> np.ndarray:
        """Pivot an object. Takes a list of lists/numpy arrays and returns the pivot."""
        array = np.stack(s, axis=1)
        rows, ridx = np.unique(array[:, self.row_idx], return_inverse=True)
        cols, cidx = np.unique(array[:, self.col_idx], return_inverse=True)
        pivot = np.zeros((len(rows), len(cols) + 1), np.double)
        pivot[ridx, cidx + 1] = array[:, self.val_idx]
        pivot[:, 0] = array[:, self.row_idx]
        return pivot


class label_binarizer_transformer(object):
    """Simple wrapper around scikit-learn's labelbinarizer."""

    def __init__(self, classlist: list[Any]):
        self.lb = LabelBinarizer().fit(classlist)

    def __call__(self, x: list[Any]) -> list[list[int]]:
        return self.lb.transform([x])


class tensor_transformer(object):
    """The simplest transformer, just returns a tensor of the input"""

    def __call__(self, x: Any) -> torch.tensor:
        return torch.as_tensor(x)


class interpolate_transformer(object):
    """Takes 0-filled data and interpolates it."""

    def __init__(self, index_col: int = 0, interp_cols: list[int] = []):
        """Creates the transformer.
        index_col is used as the point to evaluate things at.
        interp_cols are the columns that should be interpolated."""
        self.index_col = index_col
        self.interp_cols = interp_cols

    def __call__(self, obj):
        """Interpolates the data"""
        x = obj[:, self.index_col]  # store every timestamp to know what to resample to.
        for ax in self.interp_cols:
            # get non-zero values and times
            # the indicies where we have values
            nonzero_idx = obj[:, ax].nonzero()[0]
            xi = obj[nonzero_idx, self.index_col]  # the points we have
            yi = obj[nonzero_idx, ax]  # the values at those points
            obj[:, ax] = np.interp(x, xi, yi)
        return obj

class pandas_split_transformer(object):
    """Splits the input pandas series into groups by name"""

    def __init__(self, splits: list[list[Any]]):
        """Creates the transformer.
        `splits` is a list of index groups."""
        self.splits = splits

    def __call__(self, x: pd.Series) -> list[pd.Series]:
        return [x[s] for s in self.splits]
