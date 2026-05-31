#!/bin/bash
# Run the Docker container with GPU support and mount necessary directories
#
# Adjust the paths below to match your setup:
#   WORKSPACE_DIR: path to this repository on the host
#   DATA_DIR:      path to datasets and checkpoints on the host
#
# Expected directory layout inside the container:
#   /workspace/          - this repository (code)
#   /workspace/datasets/ - processed datasets (RoboCasa, Piper, etc.)
#   /workspace/checkpoints/ - model checkpoints (NVIDIA base + LoRA weights)

WORKSPACE_DIR=$(pwd)

if [ -f "${WORKSPACE_DIR}/.env" ]; then
    set -a
    source "${WORKSPACE_DIR}/.env"
    set +a
fi

DATA_DIR=${DATA_DIR:-"$(pwd)/datasets"}
CKPT_DIR=${CKPT_DIR:-"$(pwd)/checkpoints"}

docker run --shm-size=64g --gpus all -it --rm \
    --name anchordream \
    -e HF_TOKEN \
    -e WANDB_API_KEY \
    -e WANDB_ENTITY \
    -v ${WORKSPACE_DIR}:/workspace \
    -v ${DATA_DIR}:/workspace/datasets \
    -v ${CKPT_DIR}:/workspace/checkpoints \
    anchordream:latest
