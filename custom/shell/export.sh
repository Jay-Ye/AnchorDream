#!/bin/bash
# Environment setup for AnchorDream
# Run this inside the Docker container before training/inference

export PYTHONPATH=
export TOKENIZERS_PARALLELISM="false"

# Optional: WandB logging (set your own key)
# export WANDB_API_KEY="your_key_here"

# Optional: HuggingFace (for downloading checkpoints)
# export HF_TOKEN="your_token_here"
export HF_HOME=${PWD}/checkpoints
