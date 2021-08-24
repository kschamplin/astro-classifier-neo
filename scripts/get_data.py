#!/usr/bin/env python3
# downloads plasticc data and optionally preprocesses it.

from pytorch_test.plasticc import PlasticcDataModule

PlasticcDataModule(download=True).prepare_data()