# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import BaseWrapperDataset
from torch_geometric.data import Data, Batch


class PygDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def collater(self, samples):
        return Batch.from_data_list(samples)
    
    
class RepeatUnitPygsDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_to_multiple=8):
        super().__init__(dataset)
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples):
        max_repeat_num = max(len(sample) for sample in samples)   
        if self.pad_to_multiple != 1 and max_repeat_num % self.pad_to_multiple != 0:     
            max_repeat_num = int(((max_repeat_num - 0.1) // self.pad_to_multiple + 1) * self.pad_to_multiple)
        for sample_idx in range(len(samples)):
            padding_pyg = samples[sample_idx][-1].clone()
            padding_num = max_repeat_num - len(samples[sample_idx])
            samples[sample_idx] += [padding_pyg] * padding_num
        return Batch.from_data_list(sum(samples, []))    
    