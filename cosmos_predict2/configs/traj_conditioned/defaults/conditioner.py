# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from cosmos_predict2.configs.vid2vid.defaults.conditioner import Vid2VidCondition, Vid2VidConditioner
from cosmos_predict2.conditioner import T2VCondition
from cosmos_predict2.utils.context_parallel import broadcast_split_tensor


# NOTE: extend the condition class to include trajectory (latent hint)
@dataclass(frozen=True)
class TrajConditionedCondition(Vid2VidCondition):
    latent_hint: Optional[torch.Tensor] = None
    trajectory: Optional[torch.Tensor] = None
    trajectory_padding_mask: Optional[torch.Tensor] = None
    progress_bar: Optional[torch.Tensor] = None
    
    def broadcast(self, process_group: torch.distributed.ProcessGroup) -> "Vid2VidCondition":
        if self.is_broadcasted:
            return self
        # extra efforts
        gt_frames = self.gt_frames
        condition_video_input_mask_B_C_T_H_W = self.condition_video_input_mask_B_C_T_H_W
        latent_hint = self.latent_hint
        kwargs = self.to_dict(skip_underscore=False)
        kwargs["gt_frames"] = None
        kwargs["condition_video_input_mask_B_C_T_H_W"] = None
        kwargs["latent_hint"] = None
        new_condition = T2VCondition.broadcast(
            type(self)(**kwargs),
            process_group,
        )

        kwargs = new_condition.to_dict(skip_underscore=False)
        _, _, T, _, _ = gt_frames.shape
        if process_group is not None:
            if T > 1 and process_group.size() > 1:
                gt_frames = broadcast_split_tensor(gt_frames, seq_dim=2, process_group=process_group)
                condition_video_input_mask_B_C_T_H_W = broadcast_split_tensor(
                    condition_video_input_mask_B_C_T_H_W, seq_dim=2, process_group=process_group
                )
                latent_hint = broadcast_split_tensor(latent_hint, seq_dim=2, process_group=process_group)
        kwargs["gt_frames"] = gt_frames
        kwargs["condition_video_input_mask_B_C_T_H_W"] = condition_video_input_mask_B_C_T_H_W
        kwargs["latent_hint"] = latent_hint
        return type(self)(**kwargs)

# NOTE: extend the conditioner class to include trajectory (latent hint)
class TrajConditionedConditioner(Vid2VidConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> TrajConditionedCondition:
        output = super()._forward(batch, override_dropout_rate)
        assert "latent_hint" in batch, "TrajConditionedConditioner requires 'latent_hint' in batch"
        output["latent_hint"] = batch["latent_hint"]
        if "trajectory" in batch and batch["trajectory"] is not None:
            output["trajectory"] = batch["trajectory"]
            output["trajectory_padding_mask"] = batch["trajectory_padding_mask"]
        if "progress_bar" in batch and batch["progress_bar"] is not None:
            output["progress_bar"] = batch["progress_bar"]
        return TrajConditionedCondition(**output)
