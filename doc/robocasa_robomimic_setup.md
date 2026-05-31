# RoboCasa + robomimic_dev environment setup

AnchorDream's **video model** (training & observation generation) runs in the Docker
container described in the main [README](../README.md). The **simulator side** —
replaying trajectories to render robot-only (anchor) videos, and training/evaluating
imitation-learning policies — runs in a separate **conda** environment built from two
customized repositories:

- [`Jay-Ye/robocasa`](https://github.com/Jay-Ye/robocasa) — RoboCasa fork that writes
  **segmentation masks** into replay observations (needed to produce robot-only videos).
- [`Jay-Ye/robomimic_dev`](https://github.com/Jay-Ye/robomimic_dev) — robomimic fork for
  policy training and evaluation, with the AnchorDream training configs under
  `custom/training_configs/`.

You only need this environment for **Option B data preparation** (self-replay) and for
**policy learning & evaluation**. If you download our processed data and only train the
video model / generate observations, the Docker container is enough.

## Installation

```bash
# 1. Create and activate the conda environment
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa

# 2. robosuite (pinned commit)
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite && git checkout 77a4751233c29456a5381209e30dd0dbf39a6557
pip install -e .
cd ..

# 3. Customized RoboCasa (adds segmentation masks to replay observations)
git clone https://github.com/Jay-Ye/robocasa
cd robocasa
pip install -e .
python robocasa/scripts/download_kitchen_assets.py   # download kitchen assets
python robocasa/scripts/setup_macros.py              # set up local macros
cd ..

# 4. Customized robomimic (policy training & evaluation)
git clone https://github.com/Jay-Ye/robomimic_dev robomimic
cd robomimic
pip install -e .
cd ..
```

After this you will have, side by side:

```
robosuite/      # pinned robosuite
robocasa/       # customized RoboCasa (replay with segmentation)
robomimic/      # customized robomimic_dev (policy training/eval)
```

## What this environment is used for

- **Replay with segmentation** (Option B in the main README): download the original
  RoboCasa datasets and replay them to render robot-only videos —
  see [Data → Option B](../README.md#data).
- **Policy learning & evaluation**: train and evaluate policies on datasets that mix
  human demos with AnchorDream-generated observations —
  see [Policy Learning & Evaluation](../README.md#policy-learning--evaluation).
