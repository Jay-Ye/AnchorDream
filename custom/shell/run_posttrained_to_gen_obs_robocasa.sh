#!/bin/bash
# AnchorDream: Generate augmented observations for RoboCasa tasks
# This script loops over all 24 RoboCasa tasks and generates observations
# using the trained trajectory-conditioned model.
#
# Prerequisites:
# 1. Download RoboCasa HDF5 files (with robot-only videos) to datasets/raw/RoboCasa/
# 2. Download the trained checkpoint from HuggingFace
# 3. Run inside the Docker container

# Path to the trained checkpoint (download from HuggingFace)
DIT_PATH="checkpoints/anchordream-robocasa/model.pt"
CONFIG_PATH="checkpoints/anchordream-robocasa/config.yaml"

# Base directory containing RoboCasa HDF5 files
ROBOCASA_BASE_DIR="datasets/raw/RoboCasa"

# Define the list of robocasa datasets (relative to ROBOCASA_BASE_DIR)
robocasa_datasets=(
    "datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/mg/2024-05-09-09-32-19/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/mg/2024-05-04-22-35-53/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36/demo_gentex_im128_randcams_im128.hdf5"
    "datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/demo_gentex_im128_randcams_im128.hdf5"
)

# Loop over each dataset
for dataset_path in "${robocasa_datasets[@]}"; do
    echo "Processing dataset: $dataset_path"

    # Extract task name from path for unique save directory
    task_name=$(echo "$dataset_path" | grep -oP 'kitchen_\w+/\w+' | tr '/' '_')

    # Construct full local path
    full_path="${ROBOCASA_BASE_DIR}/${dataset_path}"

    # Run torchrun with current dataset
    torchrun --nproc_per_node=4 --master_port=12346 -m examples.traj_video2world_robocasa \
        --num_gpus 4 \
        --model_size 2B \
        --dit_path "${DIT_PATH}" \
        --is_lora_trained \
        --training_config_path "${CONFIG_PATH}" \
        --save_path "results/robocasa/${task_name}" \
        --disable_guardrail \
        --disable_prompt_refiner \
        --num_sampling_step 35 \
        --traj_embedder_type vanilla \
        --resolution 256 \
        --aspect_ratio 1,1 \
        --robocasa_hdf5_path "${full_path}" \
        --num_conditional_frames 5 \
        --save_suffix "vanilla_189"

    echo "Completed processing: $task_name"
    echo "----------------------------------------"
done
