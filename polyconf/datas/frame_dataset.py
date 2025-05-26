# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import yaml
import tree
import torch
import numpy as np
from easydict import EasyDict
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from openfold.utils import rigid_utils
from openfold.utils.rigid_utils import Rotation, Rigid
from data import se3_diffuser
from copy import deepcopy


def rot2trans(rigids, input_coord, key_point_index):
    frames = Rigid.from_tensor_7(deepcopy(rigids))
    rots = frames._rots.get_rot_mats()
    rotated_coord = torch.matmul(rots.unsqueeze(-3), input_coord.unsqueeze(-1)).squeeze(-1)    
    frames._trans = torch.cumsum(
        torch.cat((  
        torch.zeros((1, 3)),   
        (rotated_coord[:-1, key_point_index[3]] - rotated_coord[1:, key_point_index[0]])
        ), dim=0),
        dim=0  
    )
    return frames.to_tensor_7()
    
    
class FrameDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        with open(f'polyconf/config/base.yaml', 'r') as f:
            self.conf = EasyDict(yaml.safe_load(f))
        self.diffuser = se3_diffuser.SE3Diffuser(self.conf.diffuser)

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        rigids_0 = self.dataset[idx]["rigids_0"]
        gt_bb_rigid = rigid_utils.Rigid.from_tensor_7(rigids_0)
        
        rng = np.random.default_rng(None)

        t = rng.uniform(self.conf.data.min_t, 1.0)
        diff_feats_t = self.diffuser.forward_marginal(
            rigids_0=gt_bb_rigid,
            t=t,
            diffuse_mask=None
        )
        
        diff_feats_t['rigids_t'] = rot2trans(diff_feats_t['rigids_t'], torch.from_numpy(self.dataset[idx]["repeat_unit_atom_coordinates"]), self.dataset[idx]["key_point_index"])
        diff_feats_t['rigids_0'] = rigids_0
        diff_feats_t['t'] = t
        
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), diff_feats_t)
        
        return final_feats