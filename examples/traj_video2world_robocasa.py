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
import mediapy as mp
from tqdm import tqdm
import imageio.v2 as imageio
import glob
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
from einops import rearrange
import cv2

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.traj_conditioned.config_traj_conditioned import (
    TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B,
)
from cosmos_predict2.data.dataset_utils import uniform_subsample
from cosmos_predict2.pipelines.traj_video2world import TrajConditionedVideo2WorldPipeline
from cosmos_predict2.models.utils import load_lora_weights
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from custom.utils.utils import distributed_download_from_s3
from custom.dataset.dataset_meta import CAM_OF_INTEREST_NAME
ALL_CAMS_ROBOCASA = [cam_image.replace("_image", "") for cam_image in CAM_OF_INTEREST_NAME["RoboCasa"]]

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
        choices=["256", "480", "720"],
        default="480",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=["1,1", "4,3", "3,4", "16,9", "9,16"],
        default="3,4",
        type=str,
        help="Aspect ratio of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=[10, 16],
        default=10,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    # parser.add_argument(
    #     "--multi_view",
    #     action="store_true",
    #     help="Whether to use multi-view video for video-to-world generation",
    # )
    parser.add_argument(
        "--robocasa_hdf5_path",
        type=str,
        default="",
        help="Path to the RoboCasa HDF5 file (local path)",
    )
    parser.add_argument(
        "--traj_embedder_type",
        type=str,
        default="none",
        choices=["vanilla", "cross_attn", "none"],
        help="Type of trajectory embedder to use",
    )
    parser.add_argument(
        "--progress_emb_enabled",
        action="store_true",
        help="Whether the model is trained with progress embedder",
    )
    parser.add_argument(
        "--s3_folder",
        type=str,
        default="",
        help="S3 folder to load the model from (optional, leave empty for local paths)",
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
    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     default="",
    #     help="Text prompt for video generation",
    # )
    # parser.add_argument(
    #     "--input_video",
    #     type=str,
    #     default="assets/video2world/input3.mp4",
    #     help="Path to masked robot moving video",
    # )
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
        choices=[1, 5],
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
    parser.add_argument(
        "--save_suffix",
        type=str,
        default="",
        help="Suffix to add to the generated video key in the hdf5 file",
    )
    return parser.parse_args()

def save_multi_view_video(videos, output_path):
    # Only rank 0 should write the video
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return output_path
        
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

def parse_data_from_robocasa(episode, idx, temp_dir, trajectory_max_length=512):
    trajectory = np.concatenate([
        episode['obs']['robot0_base_to_eef_pos'],
        episode['obs']['robot0_base_to_eef_quat'],
        episode['obs']['robot0_gripper_qpos'],
    ], axis=1)
    # TODO: figure out how to handle the progress bar
    if trajectory.shape[0] > trajectory_max_length:
        log.info(f"Uniformly subsample trajectory from {trajectory.shape[0]} to {trajectory_max_length}")
        trajectory = uniform_subsample(trajectory, trajectory_max_length)
    trajectory_padded = np.zeros((1, trajectory_max_length, trajectory.shape[1]))
    trajectory_padded[0, :trajectory.shape[0], :] = trajectory
    trajectory_padding_mask = np.ones((1, trajectory_max_length))
    trajectory_padding_mask[0, :trajectory.shape[0]] = 0
    masked_videos = []
    gt_videos = []
    prompt = json.loads(episode.attrs["ep_meta"])["lang"]
    max_seg_id = episode['obs']['robot0_agentview_left_segmentation_class'][()].max()
    for camera_name in ALL_CAMS_ROBOCASA:
        segmentation = episode['obs'][f'{camera_name}_segmentation_class'][()] >= (max_seg_id-2)
        segmentation = np.flip(segmentation, axis=1) # TODO: always double check if this is correct
        videos = episode['obs'][f'{camera_name}_image'][()]
        if segmentation.shape[-1] != 1:
            segmentation = segmentation[..., None]
        gt_videos.append(videos)
        masked_videos.append(videos * segmentation)
    prompt_video_path = os.path.join(temp_dir, f"{prompt}_{idx}_masked.mp4")
    gt_video_path = os.path.join(temp_dir, f"{prompt}_{idx}_gt.mp4")
    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        save_multi_view_video(masked_videos, prompt_video_path)
        save_multi_view_video(gt_videos, gt_video_path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Wait for rank 0 to finish writing
    return trajectory_padded, trajectory_padding_mask, prompt_video_path, gt_video_path, prompt

def setup_pipeline(args: argparse.Namespace):
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config = TRAJ_CONDITIONED_PREDICT2_VIDEO2WORLD_PIPELINE_2B
        if args.traj_embedder_type == "cross_attn":
            config.net.traj_emb_config.type = "cross_attn"
            config.net.traj_emb_config.input_dim = 10 # for robocasa only, +1 for progress bar
        elif args.traj_embedder_type == "vanilla":
            config.net.traj_emb_config.type = "vanilla"
            config.net.traj_emb_config.input_dim = 10 # for robocasa only, +1 for progress bar
        elif args.traj_embedder_type == "none":
            config.net.traj_emb_config.type = "none"
        else:
            raise ValueError(f"Invalid traj_embedder_type: {args.traj_embedder_type}")
        if args.progress_emb_enabled:
            raise NotImplementedError("Progress embedder is deprecated for now")
            config.net.progress_emb_config.enabled = True
        config.resolution = args.resolution
        config.aspect_ratio = args.aspect_ratio
        # if args.fps == 10: # default is 16 so no need to change config
        #     config.state_t = 16

        dit_path = f"checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-{args.resolution}p-{args.fps}fps.pt" # TODO: change this to the post-trained model
    else:
        raise ValueError("Invalid model size. Only 2B is supported for now.")

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, 'num_gpus') and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
        
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
        load_lora_weights(pipe, dit_path, args.training_config_path, device="cuda", dtype=torch.bfloat16)
        log.success(f"Successfully loaded DiT (with lora weights) from {dit_path}")

    if hasattr(pipe.dit, "traj_embedder") and pipe.dit.traj_embedder is not None:
        log.critical("Using trajectory embedder")

    return pipe

def process_single_generation(
    pipe, input_video, video_path, prompt, output_path, negative_prompt, num_conditional_frames, num_sampling_step, guidance, seed, trajectory, trajectory_padding_mask
):
    log.info(f"Running TrajConditionedVideo2WorldPipeline\ninput: {input_video}")

    video = pipe(
        prompt=prompt,
        control_video_path=input_video,
        video_path=video_path,
        negative_prompt=negative_prompt,
        num_conditional_frames=num_conditional_frames,
        guidance=guidance,
        seed=seed,
        num_sampling_step=num_sampling_step,
        trajectory=trajectory,
        trajectory_padding_mask=trajectory_padding_mask,
    )

    if video is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        # Only rank 0 saves the video
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

def split_video_to_views(video, target_frame_size):
    """Split concatenated video into individual camera views"""
    # video is (T,H,W,C), uint8
    h, w = video.shape[1:3]
    h_target, w_target = target_frame_size
    
    # Split into 4 views
    views = []
    views.append(
        np.stack([
            cv2.resize(frame, (w_target, h_target)) for frame in video[:, :h//2, :w//2]
        ], axis=0)
    )
    views.append(
        np.stack([
            cv2.resize(frame, (w_target, h_target)) for frame in video[:, :h//2, w//2:]
        ], axis=0)
    )
    views.append(
        np.stack([
            cv2.resize(frame, (w_target, h_target)) for frame in video[:, h//2:, :w//2]
        ], axis=0)
    )
    views.append(
        np.stack([
            cv2.resize(frame, (w_target, h_target)) for frame in video[:, h//2:, w//2:]
        ], axis=0)
    )
    return views

def update_episode_in_hdf5(episode_idx, video, f, save_suffix):
    """Add generated views to HDF5 file"""
    # Only rank 0 should update the HDF5 file
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    
    # to numpy array
    if video.ndim == 5:
        video = video.squeeze(0)
    assert video.ndim == 4, "Video must have shape (T, H, W, C) or (1, T, H, W, C)"
    # Normalize to [0, 1] range
    if torch.is_floating_point(video):
        # Check if tensor is in [-1, 1] range (approximately)
        if video.min() < -0.5:
            # Convert from [-1, 1] to [0, 1]
            video = (video + 1.0) / 2.0
        video = video.clamp(0, 1)
    else:
        assert video.dtype == torch.uint8, "Only support uint8 tensor"
        video = video.float().div(255)
        
    video = (rearrange((video.cpu().float().numpy() * 255), "c t h w -> t h w c") + 0.5).astype(np.uint8)
    w_ori = video.shape[2]
    video = video[:, :, w_ori//3:w_ori*2//3] # generated video is in the middle of the original video

    episode = f["data"][episode_idx]
    # get target frame size
    target_frame_size = episode['obs'][f'{ALL_CAMS_ROBOCASA[0]}_image'].shape[1:3]

    views = split_video_to_views(video, target_frame_size)
    for view_idx, camera_name in enumerate(ALL_CAMS_ROBOCASA):
        # Create new dataset for generated view if it doesn't exist
        key = f'{camera_name}_image_generated_{save_suffix}'
        if key not in episode['obs']:
            episode['obs'].create_dataset(
                key, 
                data=views[view_idx],
                compression="gzip",
                compression_opts=6,
                shuffle=True,
                chunks=True,
            )
        else:
            episode['obs'][key][:] = views[view_idx]

def generate_video(args: argparse.Namespace, pipe: TrajConditionedVideo2WorldPipeline) -> None:
    # Only rank 0 should open the HDF5 file in write mode
    f = None
    try:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                f = h5py.File(args.robocasa_hdf5_path, "r+")  # Open in read/write mode
            else:
                f = h5py.File(args.robocasa_hdf5_path, "r")  # Open in read-only mode
        else:
            f = h5py.File(args.robocasa_hdf5_path, "r+")
    except Exception as e:
        log.error(f"Failed to open HDF5 file {args.robocasa_hdf5_path}: {e}")
        return
        
    log.info(f"Loading batch inputs from HDF5 file: {args.robocasa_hdf5_path}")    
    # Ensure file object is available
    if f is None:
        log.error("HDF5 file object is None, cannot proceed")
        return
    batch_inputs = f["data"]

    if os.environ.get("DEBUG", "0") == "1":
        log.warning("DEBUG mode, set num_sampling_step to 1")
        args.num_sampling_step = 1

    for idx, episode_idx in enumerate(tqdm(batch_inputs)):
        if f"{ALL_CAMS_ROBOCASA[0]}_image_generated_{args.save_suffix}" in batch_inputs[episode_idx]['obs']:
            log.info(f"Skipping episode {idx} as it already has generated views")
            continue

        episode = batch_inputs[episode_idx]
        trajectory, trajectory_padding_mask, prompt_video_path, gt_video_path, prompt = parse_data_from_robocasa(episode, idx, "/tmp")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        try:
            video = process_single_generation(
                pipe=pipe,
                input_video=prompt_video_path,
                video_path=gt_video_path,
                prompt=prompt, # prompt
                output_path=os.path.join(args.save_path, f"{idx}.mp4"),
                negative_prompt=args.negative_prompt,
                num_conditional_frames=args.num_conditional_frames,
                num_sampling_step=args.num_sampling_step,
                guidance=args.guidance,
                seed=args.seed,
                trajectory=trajectory,
                trajectory_padding_mask=trajectory_padding_mask,
            )

            if video is not None:
                update_episode_in_hdf5(episode_idx, video, f, args.save_suffix)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception as e:
            log.error(f"Error processing episode {idx}: {e}")
            continue
            
    # Close file only if it was opened
    if f is not None:
        try:
            f.close()
        except Exception as e:
            log.error(f"Failed to close HDF5 file: {e}")

def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    try:
        # if args.multi_view:
        #     args.input_video = save_multi_view_video(args.input_video)
        pipe = setup_pipeline(args)
        # Download S3 files after distributed environment is initialized
        if args.robocasa_hdf5_path.startswith("s3://"):
            args.robocasa_hdf5_path = distributed_download_from_s3(args.robocasa_hdf5_path, "datasets")
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()