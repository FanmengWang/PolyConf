# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import BaseWrapperDataset


def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0) :, size_w - v.size(1) :, :]
            if left_pad
            else res[i][: v.size(0), : v.size(1), :],
        )
    return res


class RightPadDatasetCoord(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False, pad_to_multiple=8):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        return collate_tokens_coords(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=self.pad_to_multiple
        )


def collate_tokens_distances(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w, size_w).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0) :, size_w - v.size(1) :, size_w - v.size(1) :]
            if left_pad
            else res[i][: v.size(0), : v.size(1), : v.size(1)],
        )
    return res


class RightPadDatasetDistance(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False, pad_to_multiple=8):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        return collate_tokens_distances(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=self.pad_to_multiple
        )
