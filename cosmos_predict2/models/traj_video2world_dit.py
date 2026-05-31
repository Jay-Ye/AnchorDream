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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import math
from typing import Literal

from einops import rearrange
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.models.text2image_dit import PatchEmbed
from cosmos_predict2.conditioner import DataType

class TrajectoryEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim=1024, num_tokens=12, depth=4, num_heads=8, max_length=512, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.input_proj = nn.Linear(input_dim, 256)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, 256))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, 256))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=num_heads, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.output_proj = nn.Linear(256, embed_dim)

    def forward(self, trajectories, padding_mask):
        """
        Args:
            trajectories: [B, max_timesteps, dimension]
            padding_mask: [B, max_timesteps] (True for padded positions, False for valid positions)
        Returns:
            tokens: [B, num_tokens, embed_dim]
        """
        B, T, _ = trajectories.shape

        x = self.input_proj(trajectories)  # [B, T, embed_dim]

        pos_embed = self.pos_embed[:, :T, :]
        x = x + pos_embed

        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # [B, T, embed_dim]

        queries = self.token_queries.expand(B, -1, -1)  # [B, num_tokens, embed_dim]
        tokens = self.transformer_decoder(
            queries, encoded, memory_key_padding_mask=padding_mask
        )  # [B, num_tokens, embed_dim]

        return self.output_proj(tokens)

class TrajEmbedderVanilla(nn.Module):
    def __init__(self, input_dim, embed_dim=1024, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)

    def forward(self, trajectories, padding_mask):
        return self.input_proj(trajectories)
    
class ProgressEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 1024, pool: Literal["mean", None] = None, mlp_dim: Optional[int] = 2048, **kwargs):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.pool = pool
        if mlp_dim is not None:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim),
            )
        else:
            self.mlp = nn.Identity()

        div_term = torch.arange(0, embed_dim //2, dtype=torch.float32)
        div_term = torch.exp(div_term * (-math.log(10000.0) / (embed_dim // 2)))
        self.register_buffer("div_term", div_term)

    def forward(self, progress_bar: torch.Tensor):
        phase = progress_bar.unsqueeze(-1) / self.div_term
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        if self.pool == "mean":
            emb = emb.mean(dim=1, keepdim=True)
        
        return self.mlp(emb)

class TrajConditionedMinimalV1LVGDiT(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        traj_emb_config = kwargs.pop("traj_emb_config")
        progress_emb_config = kwargs.pop("progress_emb_config")
        super().__init__(*args, **kwargs)

        if traj_emb_config["type"] == "cross_attn":
            self.traj_embedder = TrajectoryEmbedder(
                **traj_emb_config,
            )
        elif traj_emb_config["type"] == "vanilla":
            self.traj_embedder = TrajEmbedderVanilla(
                **traj_emb_config,
            )
        else:
            self.traj_embedder = None
        
        if progress_emb_config["enabled"]:
            self.progress_embedder = ProgressEmbedder(
                **progress_emb_config,
            )
        else:
            self.progress_embedder = None

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        latent_hint: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        trajectory: Optional[torch.Tensor] = None,
        trajectory_padding_mask: Optional[torch.Tensor] = None,
        progress_bar: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if self.traj_embedder is not None:
            traj_emb = self.traj_embedder(trajectory, trajectory_padding_mask)
            crossattn_emb = torch.cat([crossattn_emb, traj_emb], dim=1)

        if self.progress_embedder is not None:
            progress_emb = self.progress_embedder(progress_bar)
            crossattn_emb = torch.cat([crossattn_emb, progress_emb], dim=1)

        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, latent_hint.type_as(x_B_C_T_H_W)], dim=1)
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
        )


if __name__ == "__main__":
    progress_bar = torch.arange(15, 15+93) / 240
    progress_emb_config = {
        "enabled": True,
        "pool": None,
        "mlp_dim": 2048,
    }
    progress_embedder = ProgressEmbedder(**progress_emb_config)
    progress_emb = progress_embedder(progress_bar)
    print(progress_emb.shape)