# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class PrependCoordTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            padding = torch.full((item.size(0), 1, item.size(2)), self.token, dtype=item.dtype)
            item = torch.cat([padding, item], dim=1)
        return item


    