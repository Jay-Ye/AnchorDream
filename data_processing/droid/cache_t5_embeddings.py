import argparse
import json
import os
from tqdm import tqdm

from scripts.get_t5_embeddings_from_robocasa import get_t5_embeddings
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder

def main(args):
    with open(args.episode_meta_path, "r") as f:
        episode_metas = json.load(f)
    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    encoder = CosmosT5TextEncoder(device='cuda', cache_dir=args.cache_dir)

    for episode_id, episode_meta in tqdm(episode_metas.items()):
        t5_xxl_filename = os.path.join(t5_xxl_dir, f"{episode_id}.pickle")
        # if os.path.exists(t5_xxl_filename):
        #     continue
        task_descriptions = list(episode_meta["task_descriptions"].values())
        get_t5_embeddings(task_descriptions, t5_xxl_filename, args.max_length, encoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/droid_raw_subset")
    parser.add_argument("--episode_meta_path", type=str, default="datasets/droid_raw_subset/episode_meta.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default="checkpoints/google-t5/t5-11b")
    args = parser.parse_args()
    main(args)