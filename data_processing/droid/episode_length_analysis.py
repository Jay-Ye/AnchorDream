
import os
import json
import glob
from custom.dataset.dataset_meta import DATADIR
import numpy as np
from tqdm import tqdm

episode_metas = json.load(open("datasets/droid_raw_subset/episode_meta.json"))

trajectory_lengths = []
task_name_of_long_episodes = []
for episode_id, episode_meta in tqdm(episode_metas.items()):
    rel_path = episode_meta["rel_path"]
    meta_path = os.path.join(DATADIR["droid_raw"], rel_path)
    meta_file = glob.glob(os.path.join(meta_path, "metadata*.json"))
    with open(meta_file[0], "r") as f:
        meta = json.load(f)
    length = meta['trajectory_length']
    trajectory_lengths.append(length)
    
    if length > 560:
        task_name_of_long_episodes.append(episode_meta["task_descriptions"]['language_instruction1'])

print(f"mean: {np.mean(trajectory_lengths)}, std: {np.std(trajectory_lengths)}")
print(f"min: {np.min(trajectory_lengths)}, max: {np.max(trajectory_lengths)}")
print(f"median: {np.median(trajectory_lengths)}")
print(f"25th percentile: {np.percentile(trajectory_lengths, 25)}")
print(f"75th percentile: {np.percentile(trajectory_lengths, 75)}")
print(f"90th percentile: {np.percentile(trajectory_lengths, 90)}")
print(f"95th percentile: {np.percentile(trajectory_lengths, 95)}")
print(f"99th percentile: {np.percentile(trajectory_lengths, 99)}")
print(f"task names of long episodes: {'\n'.join(task_name_of_long_episodes)}")

# mean: 254.81045751633988, std: 160.4256719050596
# min: 23, max: 1973
# median: 204.0
# 25th percentile: 149.0
# 75th percentile: 310.0
# 90th percentile: 462.0
# 95th percentile: 556.0
# 99th percentile: 886.6600000000008