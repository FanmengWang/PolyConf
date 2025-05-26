# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils
from scipy.spatial.transform import Rotation as R
from torch_geometric.transforms import BaseTransform


def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v] # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):
        try:
            edge_mask, mask_rotate = data.edge_mask, data.mask_rotate
        except:
            edge_mask, mask_rotate = data.mask_edges, data.mask_rotate
            data.edge_mask = torch.tensor(data.mask_edges)

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        data.node_sigma = sigma * torch.ones(data.num_nodes)

        torsion_updates = np.random.normal(loc=0.0, scale=sigma, size=edge_mask.sum())
        data.pos = modify_conformer(data.pos, data.edge_index.T[edge_mask], mask_rotate, torsion_updates)
        data.edge_rotate = torch.tensor(torsion_updates)
        return data
    

class ConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, repeat_unit_atom_symbols, repeat_unit_atom_coordinates, repeat_unit_torsion_pyg):
        self.dataset = dataset
        self.transform = TorsionNoiseTransform()
        self.repeat_unit_atom_symbols = repeat_unit_atom_symbols
        self.repeat_unit_atom_coordinates = repeat_unit_atom_coordinates
        self.repeat_unit_torsion_pyg = repeat_unit_torsion_pyg
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        repeat_unit_atom_symbols = np.array(self.dataset[index][self.repeat_unit_atom_symbols])
        repeat_unit_atom_coordinates = self.dataset[index][self.repeat_unit_atom_coordinates]   
        repeat_unit_torsion_pygs = []
        for idx in range(repeat_unit_atom_coordinates.shape[0]):
            repeat_unit_torsion_pyg = self.dataset[index][self.repeat_unit_torsion_pyg].clone()
            repeat_unit_torsion_pyg.pos = torch.tensor((repeat_unit_atom_coordinates[idx] - repeat_unit_atom_coordinates[idx].mean(axis=0, keepdims=True)), dtype=torch.float32)
            repeat_unit_torsion_pygs.append(self.transform(repeat_unit_torsion_pyg))
        assert len(repeat_unit_atom_symbols) > 0
        assert len(repeat_unit_atom_symbols) == repeat_unit_atom_coordinates.shape[1]     
        return {"repeat_unit_atom_symbols": repeat_unit_atom_symbols,
                "repeat_unit_atom_coordinates": repeat_unit_atom_coordinates.astype(np.float32),
                "repeat_unit_torsion_pygs": repeat_unit_torsion_pygs}
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
