import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder

"""example command
python -m scripts.get_t5_embeddings_from_robocasa --dataset_path datasets/RoboCasa
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="datasets/RoboCasa", help="Root path to the dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--cache_dir", type=str, default="checkpoints/google-t5/t5-11b", help="Directory to cache the T5 model"
    )
    return parser.parse_args()


def get_t5_embeddings(prompt, t5_xxl_filename, max_length, encoder):
        # Compute T5 embeddings
        encoded_text, mask_bool = encoder.encode_prompts(
            prompt, max_length=max_length, return_mask=True
        )  # list of np.ndarray in (len, 1024)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        t5_embeddings = {}
        for task_description, t5_embedding in zip(prompt, encoded_text):
            t5_embeddings[task_description] = t5_embedding

        # Save T5 embeddings as pickle file
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(t5_embeddings, fp)

def main(args) -> None:
    tasks = os.listdir(args.dataset_path)
    # Initialize T5
    encoder = CosmosT5TextEncoder(cache_dir=args.cache_dir, local_files_only=True)
    for task in tqdm(tasks):
        meta_file = os.path.join(args.dataset_path, task, "metas", "task_description.txt")
        with open(meta_file, "r") as fp:
            prompt_list = fp.readlines()
        t5_xxl_dir = os.path.join(args.dataset_path, task, "t5_xxl")
        os.makedirs(t5_xxl_dir, exist_ok=True)
        for prompt in prompt_list:
            t5_xxl_filename = os.path.join(t5_xxl_dir, f"{prompt.strip().replace(' ', '_')}.pickle")
            if os.path.exists(t5_xxl_filename):
                # Skip if the file already exists
                continue
            get_t5_embeddings([prompt.strip()], t5_xxl_filename, args.max_length, encoder)

if __name__ == "__main__":
    args = parse_args()
    main(args)
