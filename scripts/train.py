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
import importlib
import os
import sys
import subprocess
from datetime import datetime

from loguru import logger as logging

from imaginaire.config import Config, pretty_print_overrides
from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed
from imaginaire.utils.config_helper import get_config_module, override

from custom.utils.utils import get_data_folder


####################################################
# NOTE (Dian): patch for SageMaker parsing hydra args, because SM always passes in args
# like --config1 <val1> --config2 <val2>, which is not compatible with hydra
####################################################
# add "--" after --config and --dryrun, which is expected in our case
hydra_idx = -1
if "--" not in sys.argv:
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--dryrun"):
            hydra_idx = i + 1
        if arg.startswith("--config"):
            # if passed in --config=<config>
            if "=" in arg:
                hydra_idx = i + 1
            # if passed in --config <config>
            else:
                hydra_idx = i + 2
    sys.argv.insert(hydra_idx, "--")

    # for overrides after "--", strip the "--" added by SageMaker and add "=" between hydra args
    # e.g. --config cosmos_predict2/configs/base/config.py -- --experiment <exp> becomes
    # --config cosmos_predict2/configs/base/config.py -- experiment=<exp>
    hydra_idx = sys.argv.index("--") + 1
    new_argv = sys.argv[:hydra_idx]
    for i, arg in enumerate(sys.argv[hydra_idx:]):
        if arg.startswith("--"):
            new_argv.append(arg.strip("--"))
        else:
            new_argv[-1] = new_argv[-1] + "=" + arg
    sys.argv = new_argv


def setup_s3_upload(cfg):
    if cfg.s3_folder is not None:
        job_name = os.environ.get("SM_JOB_NAME", 'local-job')
        if job_name == 'local-job':
            now = datetime.now()
            timestamp_str = now.strftime('%Y-%m-%d-%H-%M')
            job_name = job_name + f"-{timestamp_str}"

        cfg.s3_folder = cfg.s3_folder + job_name + "/"


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    # Set up unique s3 location before freezing the configs
    setup_s3_upload(config.checkpoint)
    distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Create the model
    # get_checkpoint(config.model) # Not needed anymore (it downloads models as necessary)
    model = instantiate(config.model)
    # Create the dataloaders.
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)

    # Use same name for experiment and run
    if model.recipe is not None:
        model.recipe.model.config.experiment_tag = config.job.name

    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiments=predict2_video2world_training_2b_cosmos_nemo_assets

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
