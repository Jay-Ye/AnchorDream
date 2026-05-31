#!/bin/bash

# Check if total_parts and part_idx are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <total_parts> <part_idx> <task_prefix>"
    echo "Example: $0 4 0 drawer-push"
    exit 1
fi

TOTAL_PARTS=$1
PART_IDX=$2
TASK_PREFIX=$3

# Calculate GPUs per part (8 total GPUs split across parts)
GPUS_PER_PART=$((8 / TOTAL_PARTS))
START_GPU=$((PART_IDX * GPUS_PER_PART))
END_GPU=$((START_GPU + GPUS_PER_PART - 1))

# Generate CUDA_VISIBLE_DEVICES string
GPU_LIST=""
for ((i=START_GPU; i<=END_GPU; i++)); do
    if [ $i -eq $START_GPU ]; then
        GPU_LIST="$i"
    else
        GPU_LIST="$GPU_LIST,$i"
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_LIST

echo "Running part $PART_IDX of $TOTAL_PARTS with GPUs: $GPU_LIST"

# Calculate master port to avoid conflicts between parts
MASTER_PORT=$((12346 + PART_IDX))

# Run torchrun with current dataset
torchrun --nproc_per_node=$GPUS_PER_PART --master_port=$MASTER_PORT -m examples.traj_video2world_piper \
    --num_gpus $GPUS_PER_PART \
    --model_size 2B \
    --dit_path "checkpoints/cosmos_predict2/debug/traj_cond_pred2_2b_training_piper_mul_2025-09-11_10-35-55--2025y9m11d-10h35m59s/model/iter_000021000_000542760.pt" \
    --is_lora_trained \
    --training_config_path "checkpoints/cosmos_predict2/debug/traj_cond_pred2_2b_training_piper_mul_2025-09-11_10-35-55--2025y9m11d-10h35m59s/config.yaml" \
    --save_path "results/cosmos_nemo_assets/piper" \
    --disable_guardrail \
    --disable_prompt_refiner \
    --num_sampling_step 35 \
    --traj_embedder_type vanilla \
    --resolution custom \
    --aspect_ratio 368,368 \
    --piper_aug_path "datasets/piper_data_cosmos_aug/" \
    --num_conditional_frames 5 \
    --img_save_size 180,320 \
    --state_t 22 \
    --task_prefix $TASK_PREFIX \
    --total_parts $TOTAL_PARTS \
    --part_idx $PART_IDX

echo "Completed processing part $PART_IDX"
echo "----------------------------------------"