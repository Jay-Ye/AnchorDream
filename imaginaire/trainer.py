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

import datetime
import functools
import inspect
import os
import shutil
import signal
import time

import torch
import torch.distributed as dist
import torch.utils.data

try:
    from vidar.utils.config import Config as VidarConfig
except ImportError:
    VidarConfig = None

from imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from tqdm import tqdm 

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")


from imaginaire.lazy_config import LazyConfig, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import callback, distributed, ema, log, misc
from imaginaire.utils.checkpointer import Checkpointer

try:
    from vidar.utils.config import Config as VidarConfig
except ImportError:
    VidarConfig = None

from custom.utils.utils import get_data_folder

try:
    from custom.wandb import WandbLogger
except ImportError:

    class _NoOpExperiment:
        def log(self, *args, **kwargs):
            pass

    class WandbLogger:
        """No-op fallback when custom.wandb is unavailable."""

        def __init__(self, *args, **kwargs):
            self.experiment = _NoOpExperiment()


class ImaginaireTrainer:
    """The base trainer class of Imaginaire.

    All trainers in Imaginaire should inherit ImaginaireTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        """Constructor of the trainer.

        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        super().__init__()

        # Change run name if environment variable is provided (useful for sagemaker)
        if os.getenv("JOB_NAME", None):
            config.job.__dict__['name'] = os.getenv("JOB_NAME")

        # Add datetime to the job name
        now = datetime.datetime.now()
        config.job.__dict__['name'] += f'--{now.year}y{now.month}m{now.day}d-{now.hour}h{now.minute}m{now.second}s'

        self.config = config
        # Set up the distributed computing environment.
        with misc.timer("init_distributed"):
            distributed.init()
            # Set up parallel states.
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            if USE_MEGATRON:
                if (
                    "create_gloo_process_groups"
                    in inspect.signature(parallel_state.initialize_model_parallel).parameters
                ):
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                        create_gloo_process_groups=False,
                    )
                else:
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                    )
                # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
                # It is not part of the original `parallel_state` API, so we need to set it manually.
                parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
                if parallel_state.sequence_parallel:
                    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local log.
        data_folder = get_data_folder()
        if distributed.is_rank0():
            os.makedirs(f"{data_folder}/{config.job.path_local}", exist_ok=True)
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{data_folder}/{config.job.path_local}/config.pkl")
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{data_folder}/{config.job.path_local}/config.yaml")
        dist.barrier()
        log.init_loguru_file(f"{data_folder}/{config.job.path_local}/stdout.log")
        if distributed.is_rank0():
            # Print important environment variables and the effective config.
            log.info("Config:\n" + config.pretty_print(use_color=True))
        misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # Initialize the timer for speed benchmarking.
        self.training_timer = misc.TrainingTimer()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

        # Initialize wandb logger
        if hasattr(config.model.config, 'wandb') and config.model.config.wandb['enabled'] and torch.distributed.get_rank() == 0:
            config.model.config.wandb.name = config.job.name                   # Use job name as wandb name
            config.model.config.wandb.dir = f"{data_folder}/wandb/{self.config.job.path_local}"   
            os.makedirs(config.model.config.wandb.dir, exist_ok=True)
            if VidarConfig is not None:
                self.wandb = WandbLogger(VidarConfig(**config.model.config.wandb))
            else:
                log.warning("vidar.utils.config.Config not available; wandb logging disabled.")
                self.wandb = None
        else:
            self.wandb = None

        self.time = None
        self.num_iters = 0
        self.sec_iters = 0

        self.local_path = f"{data_folder}/{self.config.job.path_local}"
        self.s3_path = None  # S3 upload disabled for local training

    def reset_logs(self, model, dataloader_train, iteration, num_samples, optimizer):
        if self.wandb is None:
            return
        wandb_logs = dict(
            global_step=iteration,
            consumed_samples=num_samples,
        )
        for key, val in optimizer.param_groups[0].items():
            if key in ['lr','weight_decay']:
                wandb_logs[key] = val
        self.time = time.time()
        self.phase = None
        return wandb_logs

    def store_logs_train(self, wandb_logs, output):
        if wandb_logs is None:
            return

        for key, val in output.items():
            if key.endswith('loss'):
                wandb_logs[f'losses/{key}'] = val.item()

        self.sec_iters = (self.sec_iters * self.num_iters + (time.time() - self.time)) / (self.num_iters + 1)
        self.num_iters += 1

        wandb_logs['iters_sec'] = (1.0 / self.sec_iters)
        wandb_logs['samples_sec'] = (1.0 / self.sec_iters) * self.samples_step
        self.phase = 'train'

        return wandb_logs

    def store_logs_val(self, wandb_logs, output):
        if wandb_logs is None:
            return

        for key, val in output.items():
            wandb_logs[key] = val
        self.phase = 'val'

        return wandb_logs

    def upload_logs(self, wandb_logs):
        if wandb_logs is None:
            return
        if self.phase is None:
            return

        self.wandb.experiment.log(wandb_logs)
        if self.phase == 'val':
            os.system(f'aws s3 sync {self.local_path}/visuals {self.s3_path}/visuals --quiet')
            os.system(f'rm {self.local_path}/visuals/*.mp4')

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration, num_samples = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        log.info("Starting training...")
        log.info(f"ckpt save frequency: {self.config.checkpoint.save_iter}")
        log.info(f"do validation: {self.config.trainer.run_validation}, validation frequency: {self.config.trainer.validation_iter}")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        self.samples_step = dataloader_train.batch_size * model.data_parallel_size
        if self.config.trainer.run_validation:
            wandb_logs = self.reset_logs(model, dataloader_train, iteration, num_samples, optimizer)
            logs = self.validate(model, dataloader_val, iteration=iteration)
            wandb_logs = self.store_logs_val(wandb_logs, logs)
            self.upload_logs(wandb_logs)
        _end_training = False
        with maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler, maybe_enable_memory_snapshot(
            self.config, global_step=iteration
        ) as memory_profiler:
            while True:
                dataloader_train_iter = iter(dataloader_train)
                progress = range(iteration, self.config.trainer.max_iter)
                if distributed.is_rank0():
                    progress = tqdm(progress, initial=iteration + 1, total=self.config.trainer.max_iter, ncols=192)
                for _ in progress:
                    num_samples += self.samples_step
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with self.training_timer("dataloader_train"):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    iteration += 1
                    # Reset wandb logs
                    wandb_logs = self.reset_logs(
                        model, dataloader_train, iteration, num_samples, optimizer)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    wandb_logs = self.store_logs_train(wandb_logs, output_batch)
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(
                            model, optimizer, scheduler, grad_scaler,
                            iteration=iteration,
                            num_samples=num_samples,
                            s3_folder=self.s3_path,
                        )
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        logs = self.validate(model, dataloader_val, iteration=iteration)
                        wandb_logs = self.store_logs_val(wandb_logs, logs)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                    # Upload wandb logs
                    self.upload_logs(wandb_logs)
                    # Update progress bar
                    if distributed.is_rank0():
                        progress.set_description(f'##### Loss: {loss.item():.5f} | Samples: {num_samples} |')
                if _end_training:
                    break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(
                model, optimizer, scheduler, grad_scaler,
                iteration=iteration,
                num_samples=num_samples,
                s3_folder=self.s3_path,
            )
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                loss_scaled.backward()
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_after_backward()
                else:
                    model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                else:
                    model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        wandb_logs = {}
        # Evaluate on the full validation set.
        with ema.ema_scope(model, enabled=model.config.pipe_config.ema.enabled):
            progress = enumerate(dataloader_val)
            if distributed.is_rank0():
                progress = tqdm(progress, ncols=192, 
                    total=len(dataloader_val), 
                    # unit_scale=model.data_parallel_size,
                )
            for val_iter, data_batch in progress:
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                output_batch, loss = model.validation_step(data_batch, iteration, self.local_path)
                if self.wandb is not None:
                    wandb_logs.update(output_batch['logs'])
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)

        self.callbacks.on_validation_end(model, iteration=iteration)
        # if self.wandb is not None and distributed.is_rank0(): # see if we still need this
        #     wandb_logs.update(
        #         {
        #             # "video": wandb.Video(
        #             #     (((output_batch[0].permute(1, 2, 3, 0)+1)/2).clamp(0, 1).cpu().numpy() * 255).astype("uint8"), fps=10, format="mp4"
        #             # )
        #             "video": wandb.Video("debug/video.mp4", format="mp4")
        #         }
        #     )
        return wandb_logs
