# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import data_utils
from unicore.data import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad, pad_to_multiple):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=self.pad_to_multiple)


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx, pad_to_multiple):
        super().__init__(dataset, pad_idx, left_pad=True, pad_to_multiple=pad_to_multiple)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx, pad_to_multiple):
        super().__init__(dataset, pad_idx, left_pad=False, pad_to_multiple=pad_to_multiple)


class RightPadDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False, pad_to_multiple=8):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_multiple = pad_to_multiple
    def collater(self, samples):
        return data_utils.collate_tokens_2d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=self.pad_to_multiple)
