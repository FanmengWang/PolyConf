# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("polyconf_phase1")
class PolyConfPhase1Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        loss = model(**sample["net_input"])
        logging_output = {
            "loss": loss.data,
            "sample_size": 1,
            "bsz": 1
        }
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
       
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train
       

@register_loss("polyconf_phase2")
class PolyConfPhase2Loss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        loss, rot_loss, dist_mat_loss = model(**sample["net_input"])
        logging_output = {
            "loss": loss.data,
            "rot_loss": rot_loss.data,
            "dist_mat_loss": dist_mat_loss.data,
            "sample_size": 1,
            "bsz": 1
        }
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        rot_loss_sum = sum(log.get("rot_loss", 0) for log in logging_outputs)
        dist_mat_loss_sum = sum(log.get("dist_mat_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "rot_loss", rot_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dist_mat_loss", dist_mat_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
       
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return is_train


@register_loss("polyconf_inference")
class PolyConfInferenceLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        logging_output = model(**sample["net_input"])
        return 1, 1, logging_output
            