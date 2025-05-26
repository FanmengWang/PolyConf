# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import torch
from unicore.data import Dictionary
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class TokenizeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary: Dictionary,
    ):
        self.dataset = dataset
        self.dictionary = dictionary

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        raw_data = self.dataset[index]
        return torch.from_numpy(self.dictionary.vec_index(raw_data)).long()