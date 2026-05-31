#!/bin/bash
# AnchorDream Training Script
# Requires: 8x NVIDIA GPUs (A100/H100 recommended), Docker container

# ==========================
# RoboCasa Training (256x256, 189 frames)
# Fine-tune Cosmos-Predict2 2B with trajectory-conditioned LoRA
# ==========================
EXP=traj_conditioned_predict2_video2world_2b_training_robocasa
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP} \
    model.config.pipe_config.net.traj_emb_config.type=vanilla \
    model.config.pipe_config.net.traj_emb_config.input_dim=10

# ==========================
# Piper Training (368x368, 93 frames)
# Example of training with custom real-world robot data
# ==========================
# EXP=traj_conditioned_predict2_video2world_2b_training_piper
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
#     --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP} \
#     model.config.pipe_config.net.traj_emb_config.type=vanilla \
#     model.config.pipe_config.net.traj_emb_config.input_dim=9 \
#     model.config.pipe_config.state_t=22
