# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.traj_conditioned.config_traj_conditioned import (
    TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
)
from cosmos_predict2.models.traj_video2world_model import (
    TrajConditionedPredict2Video2WorldModel,
)
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig, Predict2Video2WorldModelConfig
from imaginaire.lazy_config import LazyCall as L

TRAJ_CONDITIONED_PREDICT2_V2W_2B_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(TrajConditionedPredict2Video2WorldModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt",
                dit_ema_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
                vae_path="",
            ),
            fsdp_shard_size=-1,
        ),
        _recursive_=False,
    ),
)



def register_model_traj_conditioned() -> None:
    cs = ConfigStore.instance()
    # predict2 v2w 2b model
    cs.store(
        group="model",
        package="_global_",
        name="traj_conditioned_predict2_v2w_2b_fsdp",
        node=TRAJ_CONDITIONED_PREDICT2_V2W_2B_FSDP_CONFIG,
    )