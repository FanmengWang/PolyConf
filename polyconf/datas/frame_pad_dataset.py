# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import BaseWrapperDataset


def collate_rigids(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""    
    
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 7).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class RightPadDatasetRigid(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False, pad_to_multiple=8):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        return collate_rigids(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=self.pad_to_multiple
        )
        

def collate_rigid_scores(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
        
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class RightPadDatasetRigidScore(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False, pad_to_multiple=8):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        return collate_rigid_scores(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=self.pad_to_multiple
        )