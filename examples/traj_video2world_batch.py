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
from einops import rearrange

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
from custom.utils.utils import distributed_download_from_s3
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
        "--batch_input_txt",
        type=str,
        default=None,
        help="Path to a txt file that contains a list of episode paths",
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

def save_multi_view_frames(video, img_save_path):
    cams_of_interest = ALL_CAMS if "lbm" in img_save_path else ALL_CAMS_ROBOCASA
    num_frames = len(os.listdir(os.path.join(os.path.dirname(img_save_path), "rgb", cams_of_interest[0])))
    if video.ndim == 5:
        video = video[0]
    assert video.ndim == 4, "Video must have shape (C, T, H, W) or (B, C, T, H, W)"
    assert num_frames <= video.shape[1], f"Number of frames in the generated video ({video.shape[1]}) is less than the number of frames in the input video ({num_frames})"

    for cam in cams_of_interest:
        os.makedirs(os.path.join(img_save_path, cam), exist_ok=True)

    # Normalize to [0, 1] range
    if torch.is_floating_point(video):
        if video.min() < -0.5:
            video = (video + 1.0) / 2.0
        video = video.clamp(0, 1)
    else:
        assert video.dtype == torch.uint8, "Only support uint8 tensor"
        video = video.float().div(255)

    # Convert to numpy and rearrange dimensions
    frames = (rearrange((video.cpu().float().numpy() * 255), "c t h w -> t h w c") + 0.5).astype(np.uint8)

    # Each frame is a 2x2 grid, split into individual camera views
    h_orig, w_orig = frames.shape[1:3]
    frames = frames[:, :, w_orig//3:w_orig*2//3, :]
    h_new, w_new = frames.shape[1:3]

    quadrants = [
        frames[:, :h_new//2, :w_new//2],      # Top-left
        frames[:, :h_new//2, w_new//2:],      # Top-right
        frames[:, h_new//2:, :w_new//2],      # Bottom-left
        frames[:, h_new//2:, w_new//2:]       # Bottom-right
    ]

    for t in range(num_frames):
        for i, cam in enumerate(cams_of_interest):
            # Extract middle third of the quadrant
            frame = quadrants[i][t]
            imageio.imwrite(os.path.join(img_save_path, cam, f"{t:010d}.jpg"), frame)

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
            dit_path = distributed_download_from_s3(args.s3_folder + dit_path)

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

    if os.environ.get("DEBUG", "0") == "1":
        log.warning("DEBUG mode, skipping lora weights loading")
        return pipe

    if args.is_lora_trained:
        log.info(f"Loading DiT (with lora weights) from {dit_path}")
        assert os.path.exists(args.training_config_path), f"Training config file does not exist: {args.training_config_path}"
        load_lora_weights(pipe, dit_path, args.training_config_path)
        log.success(f"Successfully loaded DiT (with lora weights) from {dit_path}")

    if hasattr(pipe.dit, "traj_embedder") and pipe.dit.traj_embedder is not None:
        log.critical("Using trajectory embedder")

    return pipe

def process_single_generation(
    pipe, input_video, gt_video_path, prompt, output_path, negative_prompt, num_conditional_frames, num_sampling_step, guidance, seed, trajectory, trajectory_padding_mask
):
    log.info(f"Running TrajConditionedVideo2WorldPipeline\ninput: {input_video}")

    video = pipe(
        prompt=prompt,
        control_video_path=input_video,
        video_path=gt_video_path, # only for visualization
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
        
    return video

    #     if img_save_path is not None:
    #         save_video_frames(video, img_save_path)
    #     return True
    # return False

def generate_video(args: argparse.Namespace, pipe: TrajConditionedVideo2WorldPipeline) -> None:
    if args.batch_input_txt is not None:
        with open(args.batch_input_txt, "r") as f:
            episode_list = f.read().splitlines()
    else:
        raise NotImplementedError("Batch input txt is not supported for now")

    if os.environ.get("DEBUG", "0") == "1":
        log.warning("DEBUG mode, set num_sampling_step to 1")
        args.num_sampling_step = 1

    for idx, episode in tqdm(enumerate(episode_list), total=len(episode_list)):
        args.input_video = os.path.join(episode, "rgb_sim_rollout_masked")
        args.save_path = os.path.join("results/cosmos_nemo_assets", args.batch_input_txt.split("/")[-2], f"{episode.split('/')[-1]}.mp4")
        args.trajectory = os.path.join(episode, "lowdim", "scene_left")
        img_save_path = os.path.join(episode, "rgb_generated_processed")

        if os.path.exists(args.save_path) and os.path.exists(img_save_path):
            log.info(f"Skipping episode {idx} as it already exists: {args.save_path}")
            continue
        
        gt_video_path = os.path.join(episode, "rgb")
        try:
            gt_video_path = save_multi_view_video(gt_video_path, "/tmp/multi_view_video_gt.mp4")
        except Exception as e:
            log.error(f"Error saving gt video: {e}")
            gt_video_path = None
        if args.multi_view:
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                args.input_video = save_multi_view_video(args.input_video)
            else:
                args.input_video = "/tmp/multi_view_video.mp4"
        torch.distributed.barrier()
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

        video = process_single_generation(
            pipe=pipe,
            input_video=args.input_video,
            gt_video_path=gt_video_path,
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
        if video is not None and img_save_path is not None:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                save_multi_view_frames(video, img_save_path)
        torch.distributed.barrier()
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
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()