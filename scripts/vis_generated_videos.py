import os 
from cosmos_predict2.data.traj_conditioned.traj_conditioned_dataset import ALL_CAMS_V2 as ALL_CAMS
import cv2
import numpy as np
import imageio
from tqdm import tqdm

episode_list_file = "datasets/lbm_data_demo/BimanualStackPlatesOnTableFromDryingRack/episode_list_val.txt"

with open(episode_list_file, "r") as f:
    episode_list = f.read().splitlines()
for episode in tqdm(episode_list):
    robot_moving_video_path = os.path.join(episode, "rgb_sim_rollout_masked")
    generated_video_path = os.path.join(episode, "rgb_generated_processed")
    original_video_path = os.path.join(episode, "rgb")

    if not os.path.exists(generated_video_path):
        print(f"Generated video path does not exist: {generated_video_path}")
        continue

    num_frames = len(os.listdir(os.path.join(original_video_path, ALL_CAMS[0])))
    h, w = cv2.imread(os.path.join(original_video_path, ALL_CAMS[0], "0000000000.jpg")).shape[:2]
    video_writer = imageio.get_writer(os.path.join("debug", f"{episode.split('/')[-1]}.mp4"), fps=16, codec="libx264")
    for frame_idx in range(num_frames):
        canvas = np.zeros((h * 2, w * 2 * 3, 3), dtype=np.uint8)
        for cam_idx, cam in enumerate(ALL_CAMS):
            robot_moving_frame = cv2.resize(cv2.imread(os.path.join(robot_moving_video_path, cam, f"{frame_idx:010d}.jpg")), (w, h))
            generated_frame = cv2.resize(cv2.imread(os.path.join(generated_video_path, cam, f"{frame_idx:010d}.jpg")), (w, h))
            original_frame = cv2.resize(cv2.imread(os.path.join(original_video_path, cam, f"{frame_idx:010d}.jpg")), (w, h))
            canvas[h * (cam_idx//2):h * (cam_idx//2 + 1),  w * (cam_idx%2):w * (cam_idx%2 + 1)] = robot_moving_frame
            canvas[h * (cam_idx//2):h * (cam_idx//2 + 1),  w * (cam_idx%2 + 2):w * (cam_idx%2 + 3)] = generated_frame
            canvas[h * (cam_idx//2):h * (cam_idx//2 + 1),  w * (cam_idx%2 + 4):w * (cam_idx%2 + 5)] = original_frame
        video_writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    video_writer.close()