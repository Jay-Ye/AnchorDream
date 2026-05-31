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

from cosmos_predict2.conditioner import BooleanFlag, ReMapkey, TextAttr
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.configs.traj_conditioned.defaults.conditioner import TrajConditionedConditioner
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.traj_video2world_dit import TrajConditionedMinimalV1LVGDiT
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from imaginaire.lazy_config import LazyCall as L

from custom.utils.utils import get_data_folder
from cosmos_predict2.configs.base.config_video2world import ConditioningStrategy, CosmosReason1Config, CosmosGuardrailConfig



# Cosmos Predict2 Video2World 2B
TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_NET_2B = L(TrajConditionedMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=32, # NOTE: we simply concatenate the noisy input with the latent hint, so we need to double the in_channels
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
    # trajectory embedding settings
    traj_emb_config=dict(
        type="vanilla", # choose from cross_attn (cross attend to learnable tokens then concat), vanilla (direct concat with text embed), or none (no traj embed at all)
        input_dim=20, # dimension of the trajectory
        embed_dim=1024, # dimension of the trajectory embedding, should be the same as text embedding dimension as we'll concat them together
        num_tokens=12,
        depth=2,
        num_heads=4,
        max_length=512, # maximum length of the trajectory
    ),
    progress_emb_config=dict(
        enabled=False,
        pool=None,
        mlp_dim=2048,
        embed_dim=1024,
    )
)

TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(TrajConditionedConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
        latent_hint=L(ReMapkey)(
            input_key="latent_hint",
            output_key="latent_hint",
            dropout_rate=0.0,
            dtype=None,
        ),
        trajectory=L(ReMapkey)(
            input_key="trajectory",
            output_key="trajectory",
            dropout_rate=0.0,
            dtype=None,
        ),
        progress_bar=L(ReMapkey)(
            input_key="progress_bar",
            output_key="progress_bar",
            dropout_rate=0.0,
            dtype=None,
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=0, # this should be the number of latent frames
    max_num_conditional_frames=2, # In cases where we want to do long video generation, we have conditioning frames
    net=TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=False, # subsample the video to meet the expected length, NOT used
    resolution="480",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    control_video_key="control_video",
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=48,
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=f"{get_data_folder()}/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    ),
    # disable prompt refiner and guardrail for traj conditional
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir=f"{get_data_folder()}/checkpoints/nvidia/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=False,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=f"{get_data_folder()}/checkpoints/",
        offload_model_to_cpu=True,
        enabled=False,
    ),
)
