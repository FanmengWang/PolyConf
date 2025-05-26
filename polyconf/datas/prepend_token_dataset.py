# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class PrependTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([torch.full_like(item[0], self.token).unsqueeze(0), item], dim=0)
        return item
