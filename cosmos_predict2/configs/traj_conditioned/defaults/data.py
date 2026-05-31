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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.traj_conditioned.traj_conditioned_dataset import TrajConditionedDataset, \
    TrajConditionedDataset_droid, ConcatTrajConditionedDataset #, TrajConditionedDatasetMultiView, TrajConditionedDatasetMultiViewV2, RoboCasaDataset
from imaginaire.lazy_config import LazyCall as L

robocasa_train_dataset = L(TrajConditionedDataset)(
    dataset_name="RoboCasa",
    num_frames=189,
    video_size=(256, 256),
    phase='train',
    specified_task=None,
)

robocasa_val_dataset = L(TrajConditionedDataset)(
    dataset_name="RoboCasa",
    num_frames=189,
    video_size=(256, 256),
    phase='val',
    specified_task=None,
)

droid_train_dataset = L(TrajConditionedDataset_droid)(
    dataset_name="droid",
    num_frames=189,
    video_size=(512, 512), # TODO: change to 360, 640; the square resize looks weird in the video
    phase='train',
    specified_task=None,
    t5_index="episode",
)

droid_val_dataset = L(TrajConditionedDataset_droid)(
    dataset_name="droid",
    num_frames=189,
    video_size=(512, 512),
    phase='val',
    specified_task=None,
    t5_index="episode",
)

piper_train_dataset = L(TrajConditionedDataset)(
    dataset_name="piper",
    num_frames=93,
    video_size=(368, 368),
    phase='train',
    specified_task=None,
)

piper_val_dataset = L(TrajConditionedDataset)(
    dataset_name="piper",
    num_frames=93,
    video_size=(368, 368),
    phase='val',
    specified_task=None,
)

concat_train_dataset = L(ConcatTrajConditionedDataset)(
    datasets=[
        piper_train_dataset, 
        L(TrajConditionedDataset_droid)(
        dataset_name="droid",
        num_frames=93,
        video_size=(368, 368),
        phase='train',
        specified_task=None,
        t5_index="episode",
        two_view=True,
        euler2quat=True,
        downsample_factor=3,
        )
        ],
    weights=[0.6, 0.4],
    normalize_weights=True,
)

concat_val_dataset = L(ConcatTrajConditionedDataset)(
    datasets=[
        piper_val_dataset, 
        L(TrajConditionedDataset_droid)(
        dataset_name="droid",
        num_frames=93,
        video_size=(368, 368),
        phase='val',
        specified_task=None,
        t5_index="episode",
        two_view=True,
        euler2quat=True,
        downsample_factor=3,
        )
        ],
    weights=[0.6, 0.4],
    normalize_weights=True,
)

def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


robocasa_train_dataloader = L(DataLoader)(
    dataset=robocasa_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)

robocasa_val_dataloader = L(DataLoader)(
    dataset=robocasa_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_val_dataset),
    batch_size=1,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)

droid_train_dataloader = L(DataLoader)(
    dataset=droid_train_dataset,
    sampler=L(get_sampler)(dataset=droid_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)

droid_val_dataloader = L(DataLoader)(
    dataset=droid_val_dataset,
    sampler=L(get_sampler)(dataset=droid_val_dataset),
    batch_size=1,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)

piper_train_dataloader = L(DataLoader)(
    dataset=piper_train_dataset,
    sampler=L(get_sampler)(dataset=piper_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)

piper_val_dataloader = L(DataLoader)(
    dataset=piper_val_dataset,
    sampler=L(get_sampler)(dataset=piper_val_dataset),
    batch_size=1,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)

concat_train_dataloader = L(DataLoader)(
    dataset=concat_train_dataset,
    sampler=L(get_sampler)(dataset=concat_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)

concat_val_dataloader = L(DataLoader)(
    dataset=concat_val_dataset,
    sampler=L(get_sampler)(dataset=concat_val_dataset),
    batch_size=1,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)

def register_training_and_val_data_traj_conditioned():
    cs = ConfigStore.instance()

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="robocasa_train",
        node=robocasa_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="robocasa_val",
        node=robocasa_val_dataloader,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="droid_train",
        node=droid_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="droid_val",
        node=droid_val_dataloader,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="piper_train",
        node=piper_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="piper_val",
        node=piper_val_dataloader,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="concat_train",
        node=concat_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="concat_val",
        node=concat_val_dataloader,
    )