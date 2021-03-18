#!/usr/bin/env python

"""
    data.py
"""

import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class RaggedDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), 'len(X) != len(y)'
        self.X = [torch.LongTensor(xx) for xx in X]
        self.y = torch.LongTensor(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class SortishSampler(Sampler):
    # adapted from `fastai`
    def __init__(self, data_source, batch_size, batches_per_chunk=50):
        self.data_source = data_source
        self._key = lambda idx: len(data_source[idx])
        self.batch_size = batch_size
        self.batches_per_chunk = batches_per_chunk

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):

        idxs = np.random.permutation(len(self.data_source))

        # Group records into batches of similar size
        chunk_size = self.batch_size * self.batches_per_chunk
        chunks = [idxs[i:i + chunk_size] for i in range(0, len(idxs), chunk_size)]
        idxs = np.hstack([sorted(chunk, key=self._key, reverse=True) for chunk in chunks])

        # Make sure largest batch is in front (for memory management reasons)
        batches = [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        batch_order = np.argsort([self._key(b[0]) for b in batches])[::-1]
        batch_order[1:] = np.random.permutation(batch_order[1:])

        idxs = np.hstack([batches[i] for i in batch_order])
        return iter(idxs)


def text_collate_fn(batch, pad_value=1):
    X, y = zip(*batch)

    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]

    X = torch.stack(X, dim=-1)
    y = torch.LongTensor(y)
    return X, y
