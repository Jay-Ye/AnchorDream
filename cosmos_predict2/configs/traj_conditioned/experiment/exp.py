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

from hydra.core.config_store import ConfigStore
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig
from imaginaire.lazy_config import LazyCall as L

cs = ConfigStore.instance()

"""
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment="traj_conditional_predict2_video2world_2b_training"
"""

model_config = dict(
    config=dict(
        train_architecture="lora",
        wandb = dict(
            dir=None,
            enabled=True,
            entity='',
            project='anchordream',
            name=None, # will be set to job.name
        ),
        fsdp_shard_size=8,
        pipe_config=dict(
            ema=dict(enabled=False),
            guardrail_config=dict(enabled=False),
            prompt_refiner_config=dict(enabled=False),
        ),
    )
)

trainer_config = dict(
    distributed_parallelism="fsdp",
    callbacks=dict(
        iter_speed=dict(hit_thres=0),
    ),
    run_validation=True,
    validation_iter=250,
    max_val_iter=1, # just for debugging
)

traj_conditioned_predict2_video2world_2b_training_robocasa = dict(
    defaults=[
        {"override /model": "traj_conditioned_predict2_v2w_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_train": "robocasa_train"},
        {"override /data_val": "robocasa_val"},
    ],
    model=model_config,
    job=dict(group="debug", name="traj_cond_pred2_2b_training_robocasa_mul_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=2,
        num_workers=2,
        pin_memory=True,
    ),
    dataloader_val=dict(
        batch_size=2,
        num_workers=2,
        pin_memory=True,
    ),
    trainer=trainer_config,
    checkpoint=dict(
        save_iter=2000,
    ),
)

traj_conditioned_predict2_video2world_2b_training_droid = dict(
    defaults=[
        {"override /model": "traj_conditioned_predict2_v2w_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_train": "droid_train"},
        {"override /data_val": "droid_val"},
    ],
    model=model_config,
    job=dict(group="debug", name="traj_cond_pred2_2b_training_droid_mul_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    ),
    dataloader_val=dict(
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    ),
    trainer=trainer_config,
    checkpoint=dict(
        save_iter=500,
    ),
)

traj_conditioned_predict2_video2world_2b_training_piper = dict(
    defaults=[
        {"override /model": "traj_conditioned_predict2_v2w_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_train": "piper_train"},
        {"override /data_val": "piper_val"},
    ],
    model=model_config,
    job=dict(group="debug", name="traj_cond_pred2_2b_training_piper_mul_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=3,
        num_workers=4,
        pin_memory=True,
    ),
    dataloader_val=dict(
        batch_size=3,
        num_workers=4,
        pin_memory=True,
    ),
    trainer=trainer_config,
    checkpoint=dict(
        save_iter=500,
    ),
)

traj_conditioned_predict2_video2world_2b_training_piper_droid = dict(
    defaults=[
        {"override /model": "traj_conditioned_predict2_v2w_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_train": "concat_train"},
        {"override /data_val": "concat_val"},
    ],
    model=model_config,
    job=dict(group="debug", name="traj_cond_pred2_2b_training_piper_droid_mul_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=3,
        num_workers=4,
        pin_memory=True,
    ),
    dataloader_val=dict(
        batch_size=3,
        num_workers=4,
        pin_memory=True,
    ),
    trainer=trainer_config,
    checkpoint=dict(
        save_iter=500,
    ),
)

for _item in [
    traj_conditioned_predict2_video2world_2b_training_robocasa,
    traj_conditioned_predict2_video2world_2b_training_droid,
    traj_conditioned_predict2_video2world_2b_training_piper,
    traj_conditioned_predict2_video2world_2b_training_piper_droid,
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
