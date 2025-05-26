# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset
)

from polyconf.datas import (
    KeyDataset,
    PygDataset
)

from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("polyconf_inference")
class PolyConfInferenceTask(UnicoreTask):
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
        
        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "psmi": psmi_dataset,
                    "psmi_rep": psmi_rep_dataset,
                    "whole_pyg": PygDataset(whole_pyg_dataset),
                    "repeat_unit_smi": repeat_unit_smi_dataset, 
                    "repeat_unit_actual_num": repeat_unit_actual_num_dataset
                },
            },
        )
        
        self.datasets[split] = nest_dataset


    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model
