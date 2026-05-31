
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import os
import math
import torchvision
from einops import rearrange
from tqdm import tqdm
import imageio

from megatron.core import parallel_state

from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.conditioner import DataType, T2VCondition
from cosmos_predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2.data.dataset_utils import uniform_subsample_or_pad
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline, read_and_process_video, read_and_process_image
from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from cosmos_predict2.models.utils import load_state_dict
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.schedulers.rectified_flow_scheduler import (
    RectifiedFlowAB2Scheduler,
)
from cosmos_predict2.utils.state_dict_utils import expand_patch_embed_weights_in_state_dict
from cosmos_predict2.utils.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
)
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater
from imaginaire.utils import distributed

from custom.utils.utils import get_checkpoint, get_folder

IS_PREPROCESSED_KEY = "is_preprocessed"
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
DEFAULT_AUGMENT_SIGMA = 0.001

def read_and_process_video_full(
    video_path: str,
    resolution: list[int],
    resize: bool = True,
):
    """
    Reads a video and processes it for model input, preserving all frames.
    
    The video is loaded using easy_io and resized/cropped if specified, but keeps all frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T is the full video length.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() in _VIDEO_EXTENSIONS:
        # Load video using easy_io
        try:
            video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
            log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
        except Exception as e:
            raise ValueError(f"Failed to load video {video_path}: {e}")
    elif os.path.isdir(video_path):
        # Load video from directory
        video_frames = []
        for file in sorted(os.listdir(video_path)):
            if file.endswith(tuple(_IMAGE_EXTENSIONS)):
                image_path = os.path.join(video_path, file)
                image = imageio.imread(image_path)
                video_frames.append(image)
        assert len(video_frames) > 0, f"No images found in directory {video_path}"
        video_frames = np.array(video_frames)
        log.info(f"Loaded video from directory with shape {video_frames.shape}")
    else:
        raise ValueError(f"Invalid video path: {video_path}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    # Convert to (T, C, H, W) for resize
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)

    # Resize if needed
    if resize:
        target_h, target_w = resolution
        # Directly resize to target resolution without cropping
        video_tensor = torchvision.transforms.functional.resize(video_tensor, (target_h, target_w), antialias=True)

    # Convert to uint8
    video_tensor = (video_tensor * 255.0).to(torch.uint8)

    # Add batch dimension and permute to final format
    # [T, C, H, W] -> [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video_tensor


class TrajConditionedVideo2WorldPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

    @staticmethod
    def from_config(
        config: Video2WorldPipelineConfig,
        dit_path: str = "",
        text_encoder_path: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = TrajConditionedVideo2WorldPipeline(device=device, torch_dtype=torch_dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition)
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )

        pipe.scaling = RectifiedFlowScaling(pipe.sigma_data, config.rectified_flow_t_scaling_factor)

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert (
            pipe.tokenizer.latent_ch == pipe.config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"

        # 4. Load text encoder
        if text_encoder_path:
            # inference
            text_encoder_path = get_folder(text_encoder_path)
            pipe.text_encoder = CosmosT5TextEncoder(device=device, cache_dir=text_encoder_path)
            pipe.text_encoder.to(device)
        else:
            # training
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"

        if load_prompt_refiner:
            pipe.prompt_refiner = CosmosReason1(
                checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
                offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
                enabled=config.prompt_refiner_config.enabled,
            )

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import (
                presets as guardrail_presets,
            )

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
            pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None
            pipe.video_guardrail_runner = None

        # TODO: remove this
        if os.environ.get("DEBUG", "0") == "1":
            dit_path = None
            log.warning("DEBUG mode, skipping DiT loading")

        # 6. Set up DiT
        if dit_path:
            dit_path = get_checkpoint(dit_path)
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        # with init_weights_on_device():
        # NOTE: we don't load checkpoint on meta device since we have additional trajectory encoder
        dit_config = config.net
        pipe.dit = instantiate(dit_config).eval()  # inference

        if dit_path:
            state_dict = load_state_dict(dit_path)
            # drop net. prefix
            state_dict_dit_compatible = dict()
            for k, v in state_dict.items():
                if k.startswith("net."):
                    state_dict_dit_compatible[k[4:]] = v
                else:
                    state_dict_dit_compatible[k] = v
            if pipe.dit.x_embedder.proj[1].weight.shape[1] != state_dict_dit_compatible["x_embedder.proj.1.weight"].shape[1]:
                state_dict_dit_compatible = expand_patch_embed_weights_in_state_dict(
                    state_dict=state_dict_dit_compatible,
                    old_in_channels=state_dict_dit_compatible["x_embedder.proj.1.weight"].shape[1] // (pipe.dit.x_embedder.spatial_patch_size ** 2 * pipe.dit.x_embedder.temporal_patch_size),
                    new_in_channels=pipe.dit.x_embedder.proj[1].weight.shape[1] // (pipe.dit.x_embedder.spatial_patch_size ** 2 * pipe.dit.x_embedder.temporal_patch_size),
                    spatial_patch_size=pipe.dit.x_embedder.spatial_patch_size,
                    temporal_patch_size=pipe.dit.x_embedder.temporal_patch_size,
                    weight_key="x_embedder.proj.1.weight",
                )
                log.info(f"Expanded patch embedder weights to handle control video")
            # simply remove the traj_embedder.input_proj.weight if it's not the same shape
            # TODO: it's awkward to put it here, find a better way to handle this
            w = state_dict_dit_compatible.get("traj_embedder.input_proj.weight", None)
            if w is not None and w.shape != pipe.dit.traj_embedder.input_proj.weight.shape:
                log.warning(f"Removing traj_embedder.input_proj.weight because it's not the same shape")
                state_dict_dit_compatible.pop("traj_embedder.input_proj.weight", None)
            pipe.model_weights_with_lora = any('lora_' in k for k in state_dict_dit_compatible)
            pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            del state_dict, state_dict_dit_compatible
            log.success(f"Successfully loaded DiT from {dit_path}")

        # 6-2. Handle EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            pipe.dit_ema_worker = FastEmaModelUpdater()  # default when not using FSDP

            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            # copying is only necessary when starting the training at iteration 0.
            # Actual state_dict should be loaded after the pipe is created.
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key
        self.control_video_key = self.config.control_video_key

    def _get_data_batch_input(
        self, 
        control_video: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
        num_video_frames: int = 61,
        trajectory: Optional[np.ndarray] = None,
        trajectory_padding_mask: Optional[np.ndarray] = None,
        input_video: Optional[torch.Tensor] = None,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            negative_prompt (str): Negative prompt.
            num_latent_conditional_frames (int, optional): The number of latent conditional frames. Defaults to 1.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, _, H, W = control_video.shape
        T = num_video_frames # T is the number of video frames we generate at once. The goal is to autoregressively generate the video with the same length as the control video.

        # TODO: decide if we want to do some augmentation on the control video (e.g., blur)

        self.batch_size = 1
        data_batch = {
            "dataset_name": "video_data",
            "video": torch.zeros((1, C, T, H, W), dtype=torch.uint8).cuda() if input_video is None else input_video,
            "control_video": control_video,
            "t5_text_embeddings": self.encode_prompt(prompt).to(dtype=self.torch_dtype),
            "fps": torch.randint(16, 32, (self.batch_size,)),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),  # Padding mask (assumed no padding here)
            "num_conditional_frames": num_latent_conditional_frames,  # Specify number of conditional frames
        }
        if trajectory is not None:
            data_batch["trajectory"] = torch.from_numpy(trajectory).type_as(data_batch["t5_text_embeddings"])
            data_batch["trajectory_padding_mask"] = torch.from_numpy(trajectory_padding_mask).type_as(data_batch["t5_text_embeddings"])
        else:
            data_batch["trajectory"] = torch.zeros(1, 512, 3).type_as(data_batch["t5_text_embeddings"])
            data_batch["trajectory_padding_mask"] = torch.zeros(1, 512).type_as(data_batch["t5_text_embeddings"])

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            data_batch["neg_t5_text_embeddings"] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)
        
        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        
        return data_batch

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.
            input_key (str, optional): Key to use for input data. If None, uses self.input_data_key.

        Side Effects:
            Modifies the video tensors within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        # Determine which keys to process
        input_key = self.input_data_key if input_key is None else input_key
        input_keys = [input_key]
        if self.control_video_key in data_batch:
            input_keys.append(self.control_video_key)

        # First check if already normalized
        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            for key in input_keys:
                if key in data_batch and data_batch[key] is not None:
                    assert torch.is_floating_point(data_batch[key]), f"Video data for {key} is not in float format."
                    assert torch.all(
                        (data_batch[key] >= -1.0001) & (data_batch[key] <= 1.0001)
                    ), f"Video data for {key} is not in the range [-1, 1]. get data range [{data_batch[key].min()}, {data_batch[key].max()}]"
            return

        for key in input_keys:
            if key in data_batch and data_batch[key] is not None:
                assert data_batch[key].dtype == torch.uint8, f"Video data for {key} is not in uint8 format."
                data_batch[key] = data_batch[key].to(**self.tensor_kwargs) / 127.5 - 1.0
        data_batch[IS_PREPROCESSED_KEY] = True

        ## NOTE: should never subsample the video in transfer mode
        # # Handle temporal subsampling if needed
        # if self.config.resize_online:
        #     from torchvision.transforms.v2 import UniformTemporalSubsample

        #     expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
            
        #     # Generate subsample indices once to ensure consistent sampling across all keys
        #     subsample = UniformTemporalSubsample(expected_length)
            
        #     for key in input_keys:
        #         if key in data_batch and data_batch[key] is not None:
        #             original_length = data_batch[key].shape[2]
        #             if original_length != expected_length:
        #                 video = rearrange(data_batch[key], "b c t h w -> b t c h w")
        #                 video = subsample(video)
        #                 data_batch[key] = rearrange(video, "b t c h w -> b c t h w")

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, T2VCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        if f'{self.input_data_key}_latent' in data_batch:
            raw_state = None
            latent_state = data_batch[f'{self.input_data_key}_latent']
        else:
            raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
            latent_state = self.encode(raw_state).contiguous().float()

        if self.control_video_key in data_batch:
            control_video = data_batch[self.control_video_key]
            latent_hint = self.encode(control_video).contiguous().float()
            data_batch["latent_hint"] = latent_hint

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
        )
        return raw_state, latent_state, condition
    
    def get_x0_fn_from_batch(
            self,
            data_batch: Dict,
            guidance: float = 1.5,
            is_negative_prompt: bool = False,
            condition_latent: torch.Tensor = None,
            num_conditional_t: Union[int, None] = None,
            condition_video_augment_sigma_in_inference: float = None,
            seed: int = 1,
            # target_h: int = 54, # should be the default size for "480", check if this is correct
            # target_w: int = 96,
            # patch_h: int = 54,
            # patch_w: int = 96,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.
        
        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        - condition_latent (torch.Tensor): latent tensor in shape B,C,T,H,W as condition to generate video.
        - num_condition_t (int): number of condition latent T, used in inference to decide the condition region and config.conditioner.video_cond_bool.condition_location == "first_n"
        - condition_video_augment_sigma_in_inference (float): sigma for condition video augmentation in inference
        - seed (int, optional): The seed for the random number generator. Defaults to 1.
        - target_h (int): final stitched latent height
        - target_w (int): final stitched latent width
        - patch_h (int): latent patch height for each network inference
        - patch_w (int): latent patch width for each network inference

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        
        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        if condition_latent is None:
            condition_latent = torch.zeros(data_batch["latent_hint"].shape, **self.tensor_kwargs)

        # condition.use_video_condition should be true here
        condition = condition.set_video_condition(
            gt_frames=condition_latent, # not really gt_frames, always zeros or previous frames padded with zeros
            random_min_num_conditional_frames=self.config.min_num_conditional_frames, # not used
            random_max_num_conditional_frames=self.config.max_num_conditional_frames, # not used
            num_conditional_frames=num_conditional_t,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=condition_latent, # not really gt_frames, always zeros or previous frames padded with zeros
            random_min_num_conditional_frames=self.config.min_num_conditional_frames, # not used
            random_max_num_conditional_frames=self.config.max_num_conditional_frames, # not used
            num_conditional_frames=num_conditional_t,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_t)
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_t
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(condition_latent, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(condition_latent, uncondition, None, None)

        if not parallel_state.is_initialized():
            assert (
                not self.dit.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(
                noise_x,
                sigma,
                condition,
            ).x0 # TODO: make this x0_pred_replaced, i.e., x0 prediction with condition region replaced by gt_latent
            uncond_x0 = self.denoise(
                noise_x,
                sigma,
                uncondition,
            ).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
            return raw_x0
        
        return x0_fn
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        control_video_path: str,
        video_path: str,
        negative_prompt: str = '',
        num_conditional_frames: int = 1, # this is the number of previous frames we use during autoregressive generation
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        seed: int = 0,
        solver_option: str = "2ab",
        trajectory: Optional[np.ndarray] = None,
        trajectory_padding_mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor | None:
        # This function is only called in inference mode
        # Parameter check
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution][self.config.aspect_ratio]  # type: ignore
        height, width = self.check_resize_height_width(height, width)

        assert num_conditional_frames in [1, 5], "num_conditional_frames must be 1 or 5"
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)

        # refine prompt only if prompt refiner is enabled
        if (
            hasattr(self, "prompt_refiner")
            and self.prompt_refiner is not None
            and getattr(self.config, "prompt_refiner_config", None)
            and getattr(self.config.prompt_refiner_config, "enabled", False)
        ):
            log.info("Starting prompt refinement...")
            prompt = self.prompt_refiner.refine_prompt(video_path, prompt)
            log.info("Finished prompt refinement")

            # Run text guardrail on the refined prompt
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on refined prompt...")
                if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                    return None
                else:
                    log.success("Passed guardrail on refined prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on refined prompt are disabled")
        elif (
            hasattr(self, "config")
            and hasattr(self.config, "prompt_refiner_config")
            and not self.config.prompt_refiner_config.enabled
        ):
            log.warning("Prompt refinement is disabled")

        num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t)

        control_video = read_and_process_video_full(
            control_video_path, [height, width], resize=True
        )
        
        input_video = None
        if video_path is not None:
            input_video = read_and_process_video_full(
                video_path, [height, width], resize=True
            )

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            control_video,
            prompt,
            negative_prompt,
            num_video_frames=num_video_frames,
            trajectory=trajectory,
            trajectory_padding_mask=trajectory_padding_mask,
            input_video=input_video,
            num_latent_conditional_frames=num_latent_conditional_frames,
        )
        return self.process(data_batch, seed=seed, guidance=guidance, num_sampling_step=num_sampling_step)
    
    def process(self, data_batch, seed, guidance, num_sampling_step=10, return_latents=False, is_negative_prompt=True):
        self._normalize_video_databatch_inplace(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        n_sample = data_batch[input_key].shape[0]
        _H, _W = data_batch[input_key].shape[-2:] # the shape of the generated video in one clip
        _T = self.tokenizer.get_pixel_num_frames(self.config.state_t)
        state_shape = [
            self.config.state_ch,
            self.config.state_t,
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        ## prepare for long video generation
        control_input = data_batch[self.control_video_key]
        input_video = data_batch.get(self.input_data_key, None)
        num_conditional_frames = data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, 1) # TODO: check if this is correct
        B, C, T, H, W = control_input.shape
        num_new_generated_frames = _T - num_conditional_frames
        if (T - num_conditional_frames) % num_new_generated_frames != 0: # pad duplicate frames at the end
            pat_t = num_new_generated_frames - ((T - num_conditional_frames) % num_new_generated_frames)
            pad_frames = control_input[:, :, -1:].repeat(1, 1, pat_t, 1, 1)
            control_input = torch.cat([control_input, pad_frames], dim=2)
            num_total_frames_with_padding = control_input.shape[2]
        else:
            num_total_frames_with_padding = T
        if input_video is not None and input_video.shape[2] < num_total_frames_with_padding:
            pad_video = input_video[:, :, -1:].repeat(1, 1, num_total_frames_with_padding - input_video.shape[2], 1, 1)
            input_video = torch.cat([input_video, pad_video], dim=2)
        N_clip = (num_total_frames_with_padding - num_conditional_frames) // num_new_generated_frames

        video = []
        latents = []
        prev_frames = None
        for i_clip in range(N_clip):                
            start_frame = num_new_generated_frames * i_clip
            end_frame = num_new_generated_frames * (i_clip + 1) + num_conditional_frames
            if "progress_bar" not in data_batch:
                progress_bar = np.arange(start_frame, min(end_frame, T))
                if len(progress_bar) < end_frame - start_frame:
                    progress_bar = np.concatenate([progress_bar, progress_bar[-1:].repeat(end_frame - start_frame - len(progress_bar))])
                data_batch["progress_bar"] = torch.from_numpy(progress_bar / T).to(dtype=self.torch_dtype).unsqueeze(0) # batch size 1
            if "trajectory_indicator" not in data_batch:
                trajectory_indicator = torch.zeros(T)
                trajectory_indicator[start_frame:min(end_frame, T)] = 1
                trajectory_indicator = uniform_subsample_or_pad(trajectory_indicator, data_batch["trajectory"].shape[1])
                data_batch["trajectory"] = torch.cat([data_batch["trajectory"], trajectory_indicator[None, :, None].to(device=data_batch["trajectory"].device, dtype=data_batch["trajectory"].dtype)], dim=2)
                data_batch["trajectory_indicator"] = trajectory_indicator

            data_batch_i = {k: v for k, v in data_batch.items()}
            data_batch_i[self.control_video_key] = control_input[:, :, start_frame:end_frame].cuda()
            latent_hint = []
            log.info("Starting latent encoding")
            for b in range(B):
                data_batch_p = {k: v for k, v in data_batch_i.items()}
                data_batch_p[self.control_video_key] = data_batch_i[self.control_video_key][b:b+1]
                latent_hint.append(self.encode(data_batch_p[self.control_video_key]))
            data_batch_i["latent_hint"] = latent_hint = torch.cat(latent_hint, dim=0)
            log.info("Completed latent encoding")
            # prepare condition_latent for long video generation
            if i_clip == 0:
                _num_conditional_frames = 0 # in transfer mode, we start from no initial frames
                latent_temp = latent_hint if latent_hint.ndim == 5 else latent_hint[:, 0]
                condition_latent = torch.zeros_like(latent_temp)
                # condition_latent = self.encode(torch.ones_like(data_batch_p[self.control_video_key]).cuda() * -1) # no need to encode an empty video as it will be masked out anyway in the condition function
            else:
                _num_conditional_frames = num_conditional_frames
                input_frames = prev_frames.bfloat16().cuda() # TODO: decide if we want to clip the input frames to [-1, 1]
                condition_latent = self.encode(input_frames)

            b, c, t, h, w = condition_latent.shape
            if condition_latent.shape[2] < state_shape[1]:
                # padding condition latent to state shape
                condition_latent = torch.cat(
                    [
                        condition_latent,
                        condition_latent.new_zeros(b, c, state_shape[1] - t, h, w)
                    ],
                    dim=2,
                ).contiguous()

            x0_fn = self.get_x0_fn_from_batch( # TODO: modify this function to accept condition_latent
                data_batch_i,
                guidance=guidance,
                is_negative_prompt=is_negative_prompt,
                condition_latent=condition_latent,
                num_conditional_t=self.tokenizer.get_latent_num_frames(_num_conditional_frames),
                condition_video_augment_sigma_in_inference=DEFAULT_AUGMENT_SIGMA,
                seed=seed,
                # target_h=condition_latent.shape[-2], # TODO: check if this is correct
                # target_w=condition_latent.shape[-1], # TODO: check if this is correct
                # patch_h=condition_latent.shape[-2], # TODO: check if this is correct
                # patch_w=condition_latent.shape[-1], # TODO: check if this is correct
            )

            log.info("Starting video generation...")

            # prepare x_sigma_max
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape),
                    torch.float32,
                    self.tensor_kwargs["device"],
                    seed,
                )
                * self.scheduler.config.sigma_max
            )

            # Split the input data and condition for model parallelism, if context parallelism is enabled.
            if self.dit.is_context_parallel_enabled:
                x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

            # ------------------------------------------------------------------ #
            # Sampling loop driven by `RectifiedFlowAB2Scheduler`
            # ------------------------------------------------------------------ #
            scheduler = self.scheduler

            # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
            scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

            # Bring the initial latent into the precision expected by the scheduler
            sample = x_sigma_max.to(dtype=torch.float32)

            x0_prev: torch.Tensor | None = None

            for i, _ in enumerate(scheduler.timesteps):
                # Current noise level (sigma_t).
                sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

                # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
                # pass a 1-D tensor broadcastable to any later shape handling.
                sigma_in = sigma_t.repeat(sample.shape[0])

                # x0 prediction with conditional and unconditional branches
                x0_pred = x0_fn(sample, sigma_in)

                # Scheduler step updates the noisy sample and returns the cached x0.
                sample, x0_prev = scheduler.step(
                    x0_pred=x0_pred,
                    i=i,
                    sample=sample,
                    x0_prev=x0_prev,
                )

            # Final clean pass at sigma_min.
            sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
            sigma_in = sigma_min.repeat(sample.shape[0])
            samples = x0_fn(sample, sigma_in)

            # Merge context-parallel chunks back together if needed.
            if self.dit.is_context_parallel_enabled:
                samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

            if return_latents:
                latents.append(samples)
                continue
                
            # Decode
            video_clip = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]
            if i_clip == 0:
                video.append(video_clip)
            else:
                video.append(video_clip[:, :, num_conditional_frames:])

            prev_frames = torch.ones_like(video_clip) * -1
            prev_frames[:, :, :num_conditional_frames] = video_clip[:, :, -num_conditional_frames:]

            # # Run video guardrail on the generated video and apply postprocessing
            # if self.video_guardrail_runner is not None:
            #     # Clamp to safe range before normalization
            #     video = video.clamp(-1.0, 1.0)
            #     video_normalized = (video + 1) / 2  # [0, 1]

            #     # Convert tensor to NumPy frames for guardrail processing
            #     video_squeezed = video_normalized.squeeze(0)  # (C, T, H, W)
            #     frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            #     frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

            #     # Run guardrail
            #     processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            #     if processed_frames is None:
            #         return None
            #     else:
            #         log.success("Passed guardrail on generated video")

            #     # Convert processed frames back to tensor format
            #     processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            #     processed_video = processed_video * 2 - 1  # back to [-1, 1]
            #     processed_video = processed_video.unsqueeze(0)

            #     video = processed_video.to(video.device, dtype=video.dtype)

        if return_latents:
            return torch.cat(latents, dim=2)

        video = torch.cat(video, dim=2)[:, :, :T]
        video = torch.cat([control_input[:, :, :T], video], dim=-1)
        if input_video is not None:
            video = torch.cat([video, input_video[:, :, :T]], dim=-1)

        # if distributed.is_rank0(): # TODO: do this only during training
        #     # Save video for debugging
        #     debug_video = (video[0].permute(1, 2, 3, 0) + 1) / 2  # Convert to (T,H,W,C) and [0,1] range
        #     debug_video = (debug_video.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        #     debug_video = debug_video[:, ::3, ::3] # downsample to 1/3 size
            
        #     # Create debug directory if it doesn't exist
        #     os.makedirs("debug", exist_ok=True)
        #     debug_path = "debug/video.mp4"
        #     imageio.mimsave(debug_path, debug_video, fps=10)
        log.success("Video generation completed successfully")
        return video

