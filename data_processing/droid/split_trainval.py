import os
import random
import json
import glob
from custom.dataset.dataset_meta import DATADIR

random.seed(42)

episode_metas = json.load(open("datasets/droid_raw_subset/episode_meta.json"))
trainval_ratio = 0.95
filter_length_threshold = 560 # 95th percentile of trajectory lengths

train_episode_metas = {}
val_episode_metas = {}

episodes_by_station = {}
for episode_id, episode_meta in episode_metas.items():
    rel_path = episode_meta["rel_path"]
    meta_path = os.path.join(DATADIR["droid_raw"], rel_path)
    meta_file = glob.glob(os.path.join(meta_path, "metadata*.json"))
    with open(meta_file[0], "r") as f:
        meta = json.load(f)
    if meta['trajectory_length'] > filter_length_threshold:
        continue

    station = episode_meta["rel_path"].split("/")[0]
    if station not in episodes_by_station:
        episodes_by_station[station] = []
    episodes_by_station[station].append(episode_id)

for station, episodes in episodes_by_station.items():
    random.shuffle(episodes)
    train_episode_ids = episodes[:int(len(episodes) * trainval_ratio)]
    val_episode_ids = episodes[int(len(episodes) * trainval_ratio):]
    for episode_id in train_episode_ids:
        train_episode_metas[episode_id] = episode_metas[episode_id]
    for episode_id in val_episode_ids:
        val_episode_metas[episode_id] = episode_metas[episode_id]

with open("datasets/droid_raw_subset/episode_meta_train.json", "w") as f:
    json.dump(train_episode_metas, f, indent=4)
with open("datasets/droid_raw_subset/episode_meta_val.json", "w") as f:
    json.dump(val_episode_metas, f, indent=4)

print(f"train: {len(train_episode_metas)}, val: {len(val_episode_metas)}")