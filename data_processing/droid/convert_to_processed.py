import numpy as np
import os
import json
import h5py
import pickle
import cv2
from cosmos_predict2.data.dataset_utils import load_video
from tqdm import tqdm
import multiprocessing
from functools import partial

# DROID dataset processing

# low_dim_data_path = "demo/robocasa_demo/action"
# low_dim_data = {}
# for view in ["left", "right", "gripper"]:
#     low_dim_data[view] = []
#     for file in os.listdir(os.path.join(low_dim_data_path, view)):
#         low_dim_file_path = os.path.join(low_dim_data_path, view, file)
#         low_dim_data[view].append(np.load(low_dim_file_path, allow_pickle=True))
#     low_dim_data[view] = np.stack(low_dim_data[view])

# low_dim_data_path = "demo/lowdim"
# low_dim_data = {}
# for view in ["scene_left", "scene_right", "wrist_right_minus"]:
#     low_dim_data[view] = []
#     for file in os.listdir(os.path.join(low_dim_data_path, view)):
#         low_dim_file_path = os.path.join(low_dim_data_path, view, file)
#         low_dim_data[view].append(np.load(low_dim_file_path, allow_pickle=True))
# breakpoint()

episode_meta_path = "datasets/droid_raw_subset/episode_meta.json"
ext_data_path = "datasets/droid_raw_subset/1.0.1_extended"
data_path = "/datasets/droid_raw/droid_raw/1.0.1"
tgt_path = "datasets/Processed/droid_traj2demo/real"
# tgt_path = "datasets/Processed/droid_traj2demo_debug/real"
cam_names = ["external_cam1", "external_cam2", "wrist_cam"]
subdirs = ["lowdim", "rgb", "robot_moving_sim"]
# resize_hw = (360, 640)
resize_wh = (640, 360)

with open(episode_meta_path, "r") as f:
    episode_meta = json.load(f)

def process_episode(episode_item):
    episode_id, episode_meta_info = episode_item
    robot_moving_video_path = os.path.join(ext_data_path, episode_meta_info['rel_path'], "rgb_sim_rollout_masked") # contains 3 camera folders
    real_observation_video_path = os.path.join(data_path, episode_meta_info['rel_path'], "recordings", "MP4") # contains 3 camera mp4 files named by camera serial
    with h5py.File(os.path.join(data_path, episode_meta_info['rel_path'], "trajectory.h5"), "r") as f:
        trajectory_cartesian_position = f['observation']['robot_state']['cartesian_position'][()]
        trajectory_gripper_position = f['observation']['robot_state']['gripper_position'][()][:, None]
    with open(os.path.join(ext_data_path, "..", "t5_xxl", f"{episode_id}.pickle"), "rb") as f:
        t5_embedding = pickle.load(f)
    real_observations = {}
    robot_moving_video = {}
    for cam_name in cam_names:
        for subdir in subdirs:
            os.makedirs(os.path.join(tgt_path, episode_id, subdir, cam_name), exist_ok=True)
        real_observations[cam_name], _ = load_video(os.path.join(real_observation_video_path, episode_meta_info["cam_name_to_serial"][cam_name] + ".mp4"))
        robot_moving_video[cam_name], _ = load_video(os.path.join(robot_moving_video_path, cam_name))
    min_len = min(min(len(real_observations[cam_name]), len(robot_moving_video[cam_name])) for cam_name in cam_names)
    for t in range(min_len):
        for cam_name in cam_names:
            # Convert RGB to BGR before saving with OpenCV
            rgb_img = cv2.resize(real_observations[cam_name][t], resize_wh)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tgt_path, episode_id, "rgb", cam_name, f"{t:010}.jpg"), bgr_img)

            sim_rgb_img = cv2.resize(robot_moving_video[cam_name][t], resize_wh)
            sim_bgr_img = cv2.cvtColor(sim_rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tgt_path, episode_id, "robot_moving_sim", cam_name, f"{t:010}.jpg"), sim_bgr_img)

            embeds = [np.asarray(x) for x in t5_embedding.values()]
            proprio = {
                "cartesian_position": trajectory_cartesian_position[t:t+1],
                "gripper_position": trajectory_gripper_position[t:t+1],
            }
            np.savez(
                os.path.join(tgt_path, episode_id, "lowdim", cam_name, f"{t:010}.npz"),
                shape=resize_wh,
                proprio=np.asarray(proprio, dtype=object),
                prompt=list(t5_embedding.keys()),
                prompt_embeddings=np.asarray(embeds, dtype=object),
            )
    # print(f"processed {min_len} frames for episode {episode_id}")

episode_items = list(episode_meta.items())
# debug = 10 
# if debug > 0:
#     episode_items = episode_items[:debug]

# for episode_item in episode_items:
#     process_episode(episode_item)
#     break

with multiprocessing.Pool(processes=64) as pool: # multiprocessing.cpu_count()
    list(tqdm(pool.imap(process_episode, episode_items), total=len(episode_items)))