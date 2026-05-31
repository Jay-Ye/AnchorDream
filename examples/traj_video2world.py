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

import argparse
import json
import os
import numpy as np
import pdb
import mediapy as mp
from tqdm import tqdm
import imageio.v2 as imageio
import glob
import pickle

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.traj_conditioned.config_traj_conditioned import (
    TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
)
from cosmos_predict2.pipelines.traj_video2world import TrajConditionedVideo2WorldPipeline
from cosmos_predict2.models.utils import load_lora_weights
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from custom.utils.utils import download_from_s3
from cosmos_predict2.data.traj_conditioned.traj_conditioned_dataset import ALL_CAMS_V2 as ALL_CAMS
from cosmos_predict2.data.traj_conditioned.traj_conditioned_dataset import ALL_CAMS_ROBOCASA
from cosmos_predict2.data.dataset_utils import uniform_subsample

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--resolution",
        choices=["480", "720"],
        default="480",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=[10, 16],
        default=10,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--traj_embedder_enabled",
        action="store_true",
        help="Whether the model is trained with trajectory embedder",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="",
        help="Path to the trajectory file",
    )
    parser.add_argument(
        "--multi_view",
        action="store_true",
        help="Whether to use multi-view video for video-to-world generation",
    )
    parser.add_argument(
        "--s3_folder",
        type=str,
        default="",
        help="S3 folder to load the model from",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--is_lora_trained",
        action="store_true",
        help="Whether the model is trained with lora weights",
    )
    parser.add_argument(
        "--training_config_path",
        type=str,
        default="",
        help="Path to the training config, mandatory if is_lora_trained is True as we need to find lora configs from the training config",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="assets/video2world/input3.mp4",
        help="Path to masked robot moving video",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video-to-world generation",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning). Supposed to only be used for autoregressive generation",
    )
    parser.add_argument(
        "--num_sampling_step",
        type=int,
        default=35,
        help="Number of sampling steps for video generation",
    )
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to batch input json file",
    )
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    return parser.parse_args()

def save_multi_view_video(video_path, output_path="/tmp/multi_view_video.mp4"):
    videos_path = []
    cams_of_interest = ALL_CAMS if "lbm" in video_path else ALL_CAMS_ROBOCASA
    for cam_of_interest in cams_of_interest:
        videos_path.append(os.path.join(video_path, cam_of_interest))
    def load_video(video_path):
        image_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        frame_data = [imageio.imread(image_file) for image_file in image_files]
        frame_data = np.array(frame_data)
        return frame_data
    videos = [load_video(video_path) for video_path in videos_path]
    
    # Save each frame as a separate image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=10)

    # Create black frames if we have fewer than 4 videos
    h, w, c = videos[0][0].shape
    black_frame = np.zeros((h, w, c), dtype=np.uint8)
    # Pad videos list with black videos if needed
    while len(videos) < 4:
        videos.append(np.array([black_frame] * len(videos[0])))
        
    for t in range(len(videos[0])):
        # Concatenate the 4 views for this timestep
        top_row = np.concatenate([videos[0][t], videos[1][t]], axis=1)
        bottom_row = np.concatenate([videos[2][t], videos[3][t]], axis=1)
        frame = np.concatenate([top_row, bottom_row], axis=0)
        writer.append_data(frame)
        
    writer.close()
    return output_path

def setup_pipeline(args: argparse.Namespace):
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config = TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B
        if args.traj_embedder_enabled:
            config.net.traj_emb_config.enabled = True # TODO adapt this to the new config
            config.net.traj_emb_config.input_dim = 20 if "lbm" in args.trajectory else 9 # 20 for lbm, 9 for robocasa
        config.resolution = args.resolution
        # if args.fps == 10: # default is 16 so no need to change config
        #     config.state_t = 16

        dit_path = f"checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-{args.resolution}p-{args.fps}fps.pt" # TODO: change this to the post-trained model
    else:
        raise ValueError("Invalid model size. Only 2B is supported for now.")
    
    if hasattr(args, 'dit_path') and args.dit_path:
        dit_path = args.dit_path
        if args.s3_folder:
            dit_path = download_from_s3(args.s3_folder + dit_path)

    text_encoder_path = "checkpoints/google-t5/t5-11b"
    log.info(f"Using dit_path: {dit_path}")

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, 'num_gpus') and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    
    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False
    
    # Load models
    log.info(f"Initializing TrajConditionedVideo2WorldPipeline with model size: {args.model_size}")
    pipe = TrajConditionedVideo2WorldPipeline.from_config(
        config=config,
        dit_path=None if args.is_lora_trained else dit_path, # will load the model later if it's trained with lora weights
        text_encoder_path=text_encoder_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=False if args.disable_prompt_refiner else True,
    )
    
    if args.is_lora_trained:
        log.info(f"Loading DiT (with lora weights) from {dit_path}")
        assert os.path.exists(args.training_config_path), f"Training config file does not exist: {args.training_config_path}"
        load_lora_weights(pipe, dit_path, args.training_config_path)
        log.success(f"Successfully loaded DiT (with lora weights) from {dit_path}")

    if hasattr(pipe.dit, "traj_embedder") and pipe.dit.traj_embedder is not None:
        log.critical("Using trajectory embedder")

    return pipe

def process_single_generation(
    pipe, input_video, prompt, output_path, negative_prompt, num_conditional_frames, num_sampling_step, guidance, seed, trajectory, trajectory_padding_mask
):
    log.info(f"Running TrajConditionedVideo2WorldPipeline\ninput: {input_video}")

    video = pipe(
        prompt=prompt,
        control_video_path=input_video,
        video_path=input_video, # this should just be a placeholder TODO: pass in the real video path if we want to use prompt_refiner
        negative_prompt=negative_prompt,
        num_conditional_frames=num_conditional_frames,
        guidance=guidance,
        seed=seed,
        num_sampling_step=num_sampling_step,
        trajectory=trajectory,
        trajectory_padding_mask=trajectory_padding_mask,
    )

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        if pipe.config.state_t == 16:
            fps = 10
        else:
            fps = 16
        save_image_or_video(video, output_path, fps=fps)
        log.success(f"Successfully saved video to: {output_path}")
        return True
    return False

def generate_video(args: argparse.Namespace, pipe: TrajConditionedVideo2WorldPipeline) -> None:
    if args.batch_input_json is not None:
        raise NotImplementedError("Batch input JSON is not supported for now")
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json, "r") as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(tqdm(batch_inputs)):
            input_video = item.get("input_video", "")
            prompt = item.get("prompt", "")
            output_video = item.get("output_video", f"output_{idx}.mp4")

            if not input_video or not prompt:
                log.warning(f"Skipping item {idx}: Missing input_video or prompt")
                continue

            process_single_generation(
                pipe=pipe,
                input_video=input_video,
                prompt=prompt,
                output_path=output_video,
                negative_prompt=args.negative_prompt,
                num_conditional_frames=args.num_conditional_frames,
                num_sampling_step=args.num_sampling_step,
                guidance=args.guidance,
                seed=args.seed,
            )
    else:
        trajectory_padded = None
        trajectory_padding_mask = None
        if args.traj_embedder_enabled:

            if "lbm" in args.trajectory:
                trajectory = []
                timestep_files = sorted(glob.glob(os.path.join(args.trajectory, "*.npz")))
                for timestep_file in timestep_files:
                    proprio = np.load(timestep_file, allow_pickle=True)['proprio'].item()
                    trajectory.append(
                        np.concatenate([
                            proprio['robot__actual__poses__right::panda__xyz'][0],
                            proprio['robot__actual__poses__right::panda__rot_6d'][0],
                            proprio['robot__actual__poses__left::panda__xyz'][0],
                            proprio['robot__actual__poses__left::panda__rot_6d'][0],
                            proprio['robot__actual__grippers__right::panda_hand'][0],
                            proprio['robot__actual__grippers__left::panda_hand'][0],
                        ], axis=0)
                    )
                trajectory = np.stack(trajectory, axis=0)
            else:
                with open(args.trajectory, "rb") as f:
                    trajectory = pickle.load(f)
                trajectory = np.concatenate([
                    trajectory['robot0_base_to_eef_pos'],
                    trajectory['robot0_base_to_eef_quat'],
                    trajectory['robot0_gripper_qpos'],
                ], axis=1)

            if trajectory.shape[0] > 512:
                log.info(f"Uniformly sampling trajectory from {trajectory.shape[0]} to 512")
                trajectory = uniform_subsample(trajectory, 512)

            trajectory_padded = np.zeros((1, 512, trajectory.shape[1]))
            trajectory_padded[0, :trajectory.shape[0]] = trajectory
            trajectory_padding_mask = np.ones((1, 512)).astype(bool)
            trajectory_padding_mask[0, :trajectory.shape[0]] = 0

        process_single_generation(
            pipe=pipe,
            input_video=args.input_video,
            prompt=args.prompt,
            output_path=args.save_path,
            negative_prompt=args.negative_prompt,
            num_conditional_frames=args.num_conditional_frames,
            num_sampling_step=args.num_sampling_step,
            guidance=args.guidance,
            seed=args.seed,
            trajectory=trajectory_padded,
            trajectory_padding_mask=trajectory_padding_mask,
        )
    return

def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    try:
        if args.multi_view:
            args.input_video = save_multi_view_video(args.input_video)
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()