
import os
import json
import glob
from custom.dataset.dataset_meta import DATADIR
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

dataset_name = 'piper'
dataset_dir = DATADIR[dataset_name]

tasks = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d not in ["t5_xxl", "metas"]]
episode_paths = []
for task in tasks:
    with open(os.path.join(os.path.join(dataset_dir, task, "episode_list.txt")), "r") as f:
        episode_list = f.read().splitlines()
    for episode in episode_list:
        episode_paths.append({
            "task": task,
            "path": os.path.join(dataset_dir, task, episode),
            "trajectory_path": os.path.join(dataset_dir, task, episode, "lowdim", "lowdim.pkl"),
        })

trajectory_lengths = []
task_name_of_long_episodes = []
for episode_meta in tqdm(episode_paths):
    with open(episode_meta['trajectory_path'], "rb") as f:
        trajectory = pickle.load(f)
    trajectory_lengths.append(trajectory['sim_replay_end_pose'].shape[0])

print(f"mean: {np.mean(trajectory_lengths)}, std: {np.std(trajectory_lengths)}")
print(f"min: {np.min(trajectory_lengths)}, max: {np.max(trajectory_lengths)}")
print(f"median: {np.median(trajectory_lengths)}")
print(f"25th percentile: {np.percentile(trajectory_lengths, 25)}")
print(f"75th percentile: {np.percentile(trajectory_lengths, 75)}")
print(f"90th percentile: {np.percentile(trajectory_lengths, 90)}")
print(f"95th percentile: {np.percentile(trajectory_lengths, 95)}")
print(f"99th percentile: {np.percentile(trajectory_lengths, 99)}")

# Create histogram
plt.figure(figsize=(12, 8))
plt.hist(trajectory_lengths, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Trajectory Length')
plt.ylabel('Frequency')
plt.title('Distribution of Trajectory Lengths')
plt.grid(True, alpha=0.3)

# Add vertical lines for statistics
plt.axvline(np.mean(trajectory_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(trajectory_lengths):.1f}')
plt.axvline(np.median(trajectory_lengths), color='orange', linestyle='--', label=f'Median: {np.median(trajectory_lengths):.1f}')
plt.axvline(np.percentile(trajectory_lengths, 95), color='purple', linestyle='--', label=f'95th percentile: {np.percentile(trajectory_lengths, 95):.1f}')

plt.legend()
plt.tight_layout()
plt.savefig('trajectory_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# mean: 254.81045751633988, std: 160.4256719050596
# min: 23, max: 1973
# median: 204.0
# 25th percentile: 149.0
# 75th percentile: 310.0
# 90th percentile: 462.0
# 95th percentile: 556.0
# 99th percentile: 886.6600000000008