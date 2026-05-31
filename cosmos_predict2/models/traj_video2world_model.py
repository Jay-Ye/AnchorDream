import math
import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from cosmos_predict2.pipelines.traj_video2world import TrajConditionedVideo2WorldPipeline
from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from cosmos_predict2.models.utils import assign_lora_weights

class TrajConditionedPredict2Video2WorldModel(Predict2Video2WorldModel):
    def __init__(self, config: Predict2Video2WorldModelConfig):
        super(ImaginaireModel, self).__init__()
        # New code, added for i4 adaption
        learning_rate = config.learning_rate

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")

        # Set up different modules
        self.setup_modules(config)

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        self.pipe = TrajConditionedVideo2WorldPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
            text_encoder_path=config.model_manager_config.text_encoder_path,
        )

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.model_weights_with_lora:
                log.critical(f"(Re) Loading DiT (with lora weights) from {config.model_manager_config.dit_path}")
                assign_lora_weights(self.pipe.dit, config.model_manager_config.dit_path)
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
                if self.pipe.model_weights_with_lora:
                    log.critical(f"(Re) Loading DiT ema (with lora weights) from {config.model_manager_config.dit_path}")
                    assign_lora_weights(self.pipe.dit_ema, config.model_manager_config.dit_path)
        else:
            self.pipe.denoising_model().requires_grad_(True)
        self.pipe.dit.x_embedder.requires_grad_(True) # NOTE: we need to train the x_embedder since we have additional channels
        if self.pipe.dit.traj_embedder is not None:
            log.critical("Training trajectory embedder")
            self.pipe.dit.traj_embedder.requires_grad_(True)
        if self.pipe.dit.progress_embedder is not None:
            log.critical("Training progress embedder")
            self.pipe.dit.progress_embedder.requires_grad_(True)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

        self.learning_rate = learning_rate