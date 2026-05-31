import os
import h5py
import numpy as np
import argparse
import json
import pickle
from PIL import Image
from metadata import HDF5_FILES
import imageio
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def save_imgs(data, output_dir):
    for i, img in enumerate(data):
        img_path = os.path.join(output_dir, f"{i:010d}.jpg")
        Image.fromarray(img).save(img_path)

def save_multiview_video(data, output_dir):
    writer = imageio.get_writer(output_dir, fps=10)
    
    h, w, c = data[0][0].shape
    black_frame = np.zeros((h, w, c), dtype=np.uint8)
    while len(data) < 4:
        data.append(np.array([black_frame] * len(data[0])))
        
    for t in range(len(data[0])):
        top_row = np.concatenate([data[0][t], data[1][t]], axis=1)
        bottom_row = np.concatenate([data[2][t], data[3][t]], axis=1)
        frame = np.concatenate([top_row, bottom_row], axis=0)
        writer.append_data(frame)
        
    writer.close()

def save_lowdim(data, output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "wb") as f:
        pickle.dump(data, f)

def save_control_video(img_obs, seg_obs, output_dir):
    max_seg_id = max(singleview.max() for singleview in seg_obs.values())
    masked_videos = []
    for key in img_obs.keys():
        segmentation = seg_obs[key.replace("image", "segmentation_class")] >= (max_seg_id-2)
        # Flip segmentation upside down
        segmentation = np.flip(segmentation, axis=1)
        masked_video = img_obs[key] * segmentation
        masked_videos.append(masked_video)
    os.makedirs(output_dir, exist_ok=True)
    multiview_video_path = os.path.join(output_dir, "multiview_video.mp4")
    save_multiview_video(masked_videos, multiview_video_path)

def save_gt_video(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    multiview_video_path = os.path.join(output_dir, "multiview_video.mp4")
    save_multiview_video(list(data.values()), multiview_video_path)

def process_hdf5_file(hdf5_file, local_base_dir, output_dir):
    hdf5_file = os.path.join(local_base_dir, hdf5_file)
    task_name = hdf5_file.split("/")[-3]

    if not os.path.exists(hdf5_file):
        print(f"[WARNING] Skipping {task_name}: HDF5 file not found at {hdf5_file}")
        return

    task_dir = os.path.join(output_dir, task_name)
    data_type = "human" # "human" or "mg"

    os.makedirs(os.path.join(task_dir, "metas"), exist_ok=True)

    episode_dict = {} # maintain a dict of {episode_dir: task_description}
    with h5py.File(hdf5_file, "r") as f:
        for episode_id in tqdm(f["data"].keys(), desc=f"Processing {task_name}"):
            episode_dir = os.path.join(task_dir, data_type, episode_id)
            os.makedirs(episode_dir, exist_ok=True)
            task_description = json.loads(f["data"][episode_id].attrs["ep_meta"])["lang"]
            episode_dict[episode_dir] = task_description
            trajectory = {
                "robot0_base_to_eef_pos": f["data"][episode_id]["obs"]["robot0_base_to_eef_pos"][()],
                "robot0_base_to_eef_quat": f["data"][episode_id]["obs"]["robot0_base_to_eef_quat"][()],
                "robot0_gripper_qpos": f["data"][episode_id]["obs"]["robot0_gripper_qpos"][()],
                "task_description": task_description,
            }
            img_obs = {
                "robot0_agentview_left_image": f["data"][episode_id]["obs"]["robot0_agentview_left_image"][()],
                "robot0_agentview_right_image": f["data"][episode_id]["obs"]["robot0_agentview_right_image"][()],
                "robot0_eye_in_hand_image": f["data"][episode_id]["obs"]["robot0_eye_in_hand_image"][()],
            }
            seg_obs = {
                "robot0_agentview_left_segmentation_class": f["data"][episode_id]["obs"]["robot0_agentview_left_segmentation_class"][()],
                "robot0_agentview_right_segmentation_class": f["data"][episode_id]["obs"]["robot0_agentview_right_segmentation_class"][()],
                "robot0_eye_in_hand_segmentation_class": f["data"][episode_id]["obs"]["robot0_eye_in_hand_segmentation_class"][()],
            }
            save_lowdim(trajectory, os.path.join(episode_dir, "lowdim", "lowdim.pkl"))
            save_gt_video(img_obs, os.path.join(episode_dir, "rgb"))
            save_control_video(img_obs, seg_obs, os.path.join(episode_dir, "rgb_sim_rollout_masked"))
            
    # save unique task descriptions to a txt file
    unique_task_descriptions = list(set(episode_dict.values()))
    with open(os.path.join(task_dir, "metas", "task_description.txt"), "w") as f:
        for task_description in unique_task_descriptions:
            f.write(task_description + "\n")

    episode_list = list(episode_dict.keys())
    with open(os.path.join(task_dir, "episode_list.txt"), "w") as f:
        for episode_dir in episode_list:
            f.write(episode_dir.replace(task_dir+"/", "") + "\n")

def main(args):
    # Create a pool of workers
    pool = mp.Pool(processes=mp.cpu_count())
    
    # Create partial function with fixed arguments
    process_file = partial(process_hdf5_file, 
                         local_base_dir=args.local_base_dir,
                         output_dir=args.output_dir)
    
    # Process files in parallel
    list(tqdm(pool.imap(process_file, HDF5_FILES), total=len(HDF5_FILES)))
    
    # Close the pool
    pool.close()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_base_dir", type=str, default="datasets/raw/RoboCasa",
                        help="The base directory of the processed hdf5 files")
    parser.add_argument("--output_dir", type=str, default="datasets/RoboCasa",
                        help="The base directory of the output dataset")
    args = parser.parse_args()

    main(args)