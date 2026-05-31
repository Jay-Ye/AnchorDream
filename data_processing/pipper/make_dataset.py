import argparse
import os
from scripts.get_t5_embeddings_from_robocasa import get_t5_embeddings
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder

task_descriptions = {
    "book": ["put the book on the bookshelf", "place the book on the shelf"],
    "drawer-pull": ["pull the drawer open", "open the drawer"],
    "drawer-push": ["push the drawer closed", "close the drawer"],
    "pnp": ["pick up the toy and put it into the bowl", "grasp the toy and place it in the bowl", "pick up the toy and drop it into the bowl"],
    "pour": ["pick up the cup and pour it into the larger bowl", "grasp the cup and tilt it to pour its contents into the bowl", "lift the cup and pour into the bowl"],
    "sweep": ["grasp the brush by its handle and use it to sweep the coffee beans toward the left side of the table", "sweep the coffee beans toward left with the brush"],
}

def main(args):
    task_folders = os.listdir(args.dataset_dir)
    text_encoder = CosmosT5TextEncoder(device='cuda', cache_dir=args.cache_dir)
    for task_folder in task_folders:
        task_folder_path = os.path.join(args.dataset_dir, task_folder)

        for task in task_descriptions:
            if task_folder.startswith(task):
                task_description = task_descriptions[task]

        metas_dir = os.path.join(task_folder_path, "metas")
        os.makedirs(metas_dir, exist_ok=True)
        with open(os.path.join(metas_dir, "task_description.txt"), "w") as f:
            for prompt in task_description:
                f.write(prompt + "\n")

        t5_xxl_dir = os.path.join(task_folder_path, "t5_xxl")
        os.makedirs(t5_xxl_dir, exist_ok=True)
        for prompt in task_description:
            t5_xxl_filename = os.path.join(t5_xxl_dir, f"{prompt.strip().replace(' ', '_')}.pickle")
            if os.path.exists(t5_xxl_filename):
                continue
            get_t5_embeddings([prompt.strip()], t5_xxl_filename, args.max_length, text_encoder)

        episode_folders = [episode for episode in os.listdir(task_folder_path) if os.path.isdir(os.path.join(task_folder_path, episode)) and "rgb" in os.listdir(os.path.join(task_folder_path, episode))]
        with open(os.path.join(task_folder_path, "episode_list.txt"), "w") as f:
            for episode_folder in episode_folders:
                f.write(episode_folder + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="checkpoints/google-t5/t5-11b")
    parser.add_argument("--dataset_dir", type=str, default="datasets/piper_data_cosmos")
    parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the T5 embeddings")
    args = parser.parse_args()

    main(args)