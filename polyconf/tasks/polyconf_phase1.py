# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    SortDataset,
    FromNumpyDataset,
)

from polyconf.datas import (
    KeyDataset,
    TokenizeDataset,
    ConformerDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDataset,
    RightPadDataset2D,
    RightPadDatasetCoord,
    RightPadDatasetDistance,
    data_utils,
    AppendTokenDataset,
    PrependTokenDataset,
    PrependCoordTokenDataset,
    AppendCoordTokenDataset,
    PygDataset,
    RepeatUnitPygsDataset,
)

from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("polyconf_phase1")
class PolyConfPhase1Task(UnicoreTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument("--dict-name", default="dict.txt", help="dictionary file")
        parser.add_argument("--pad-to-multiple", type=int, default=8, help="padding alignment size")

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        
        psmi_dataset = KeyDataset(dataset, "psmi") # (bsz, )
        psmi_rep_dataset = KeyDataset(dataset, "psmi_rep") # (bsz, 768)
        whole_pyg_dataset = KeyDataset(dataset, "whole_pyg") # (bsz, )
        
        repeat_unit_smi_dataset = KeyDataset(dataset, "repeat_unit_smi") # (bsz, )
        repeat_unit_actual_num_dataset = KeyDataset(dataset, "repeat_unit_actual_num") # (bsz, )
        
        dataset = ConformerDataset(
                dataset, "repeat_unit_atom_symbols", "repeat_unit_atom_coordinates", "repeat_unit_torsion_pyg"
            )
        dataset = NormalizeDataset(dataset, "repeat_unit_atom_coordinates", normalize_coord=True)
        
        repeat_unit_torsion_pygs_dataset = KeyDataset(dataset, "repeat_unit_torsion_pygs") # (bsz, r_n)
        
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = KeyDataset(dataset, "repeat_unit_atom_symbols")
        src_dataset = TokenizeDataset(src_dataset, self.dictionary)
        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos()) # (bsz, n)
        
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary)) # (bsz, n, n)
        
        def PrependAndAppendCoord(dataset, pre_token, app_token):
            dataset = PrependCoordTokenDataset(dataset, pre_token)
            return AppendCoordTokenDataset(dataset, app_token)
                
        coord_dataset = KeyDataset(dataset, "repeat_unit_atom_coordinates")
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppendCoord(coord_dataset, 0.0, 0.0) # (bsz, r_n, n, 3)
                
        distance_dataset = DistanceDataset(coord_dataset) # (bsz, r_n, n, n)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "psmi": psmi_dataset,
                    "psmi_rep": psmi_rep_dataset,
                    "whole_pyg": PygDataset(whole_pyg_dataset), 
                    "repeat_unit_smi": repeat_unit_smi_dataset,
                    "repeat_unit_actual_num": repeat_unit_actual_num_dataset,
                    "repeat_unit_torsion_pygs": RepeatUnitPygsDataset(repeat_unit_torsion_pygs_dataset, pad_to_multiple=self.args.pad_to_multiple),
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                    "src_distance": RightPadDatasetDistance(
                        distance_dataset,
                        pad_idx=0,
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                },
            },
        )
        
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))
            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model
