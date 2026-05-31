import argparse
import os
import json
from metadata import CAM2IMAGE_MAP, IMAGE2SCENE_CAM_MAP
from cosmos_predict2.data.dataset_utils import load_video
import numpy as np
import imageio
import h5py
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import cv2

def get_spatially_concatenated_video(view_paths, output_path, fps=None, HW=(720, 1280)):
    """
    view_paths: list of paths to the views (mp4 files or jpg folders) to be spatially concatenated
    output_path: path to save the spatially concatenated video
    """
    frame_data_list = []
    for view_path in view_paths:
        frame_data, _fps = load_video(view_path)
        frame_data_list.append(frame_data)

    if fps is None:
        fps = _fps

    T_min = np.min([frame_data.shape[0] for frame_data in frame_data_list])
    for i, frame_data in enumerate(frame_data_list):
        T, H, W, C = frame_data.shape
        if T > T_min:
            print(f"Warning: {view_paths[i]} has {T} frames, but the minimum number of frames is {T_min}")
            frame_data_list[i] = frame_data[:T_min]
        if H != HW[0] or W != HW[1]:
            resized_frames = np.zeros((T, HW[0], HW[1], C), dtype=frame_data.dtype)
            for t in range(T):
                resized_frames[t] = cv2.resize(frame_data[t], (HW[1], HW[0]))
            frame_data_list[i] = resized_frames
    if len(frame_data_list) < 4:
        # pad with a black video if the number of cameras is less than 4
        frame_data_list = frame_data_list + [np.zeros_like(frame_data_list[0])] * (4 - len(frame_data_list))
    top_row = np.concatenate([frame_data_list[0], frame_data_list[1]], axis=2)
    bottom_row = np.concatenate([frame_data_list[2], frame_data_list[3]], axis=2)
    frames = np.concatenate([top_row, bottom_row], axis=1)
    imageio.mimsave(output_path, frames, fps=fps)

def process_episode_from_item(item):
    args, (episode_id, episode_meta) = item
    return process_episode(args, episode_id, episode_meta)

def process_episode(args, episode_id, episode_meta):
    try:
        input_view_paths = []
        gt_view_paths = []
        for cam_name in IMAGE2SCENE_CAM_MAP.values():
            input_path = os.path.join(args.local_base_dir, episode_meta["rel_path"], args.replay_folder, cam_name)
            gt_path = os.path.join(args.droid_raw_dir, episode_meta["rel_path"], "recordings", "MP4", episode_meta["cam_name_to_serial"][cam_name] + ".mp4")
            input_view_paths.append(input_path)
            gt_view_paths.append(gt_path)

        get_spatially_concatenated_video(input_view_paths, os.path.join(args.local_base_dir, episode_meta["rel_path"], "input_multiview_video.mp4"), fps=60.0)
        get_spatially_concatenated_video(gt_view_paths, os.path.join(args.local_base_dir, episode_meta["rel_path"], "gt_multiview_video.mp4"), fps=60.0)
        return None
    except Exception as e:
        print(f"Error processing episode {episode_id}: {str(e)}")
        return (episode_id, str(e))

def main(args):
    episode_id_to_path = json.load(open(os.path.join(args.path_to_droid_meta, "episode_id_to_path.json")))
    episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}
    episode_id_to_task_description = json.load(open(os.path.join(args.path_to_droid_meta, "droid_language_annotations.json")))
    episode_id_to_cam_to_serial = json.load(open(os.path.join(args.path_to_droid_meta, "camera_serials.json")))

    episode_metas = {}
    episode_without_task_description = []
    for station in os.listdir(args.local_base_dir):
        station_dir = os.path.join(args.local_base_dir, station, "success")
        for date in os.listdir(station_dir):
            date_dir = os.path.join(station_dir, date)
            for episode in os.listdir(date_dir):
                episode_dir = os.path.join(date_dir, episode)
                rel_path = os.path.relpath(episode_dir, args.local_base_dir)
                try:
                    episode_id = episode_path_to_id[rel_path]
                except:
                    episode_id = episode_path_to_id[rel_path.replace(":", "_")]
                try:
                    episode_metas[episode_id] = {
                        "rel_path": rel_path,
                        "task_descriptions": episode_id_to_task_description[episode_id], # NOTE: there are ~385 episodes without task description, figure out why
                        "cam_name_to_serial": {
                            IMAGE2SCENE_CAM_MAP[CAM2IMAGE_MAP[cam_name]]: serial
                            for cam_name, serial in episode_id_to_cam_to_serial[episode_id].items()
                        }
                    }
                except:
                    episode_without_task_description.append(episode_id)
                    continue
    print(f"Number of episodes without task description: {len(episode_without_task_description)}")
    with open(os.path.join(os.path.dirname(args.local_base_dir), "episode_meta.json"), "w") as f:
        json.dump(episode_metas, f, indent=4)

    if args.num_workers > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            iterable = ((args, kv) for kv in episode_metas.items())  # kv is (episode_id, episode_meta)
            results = []
            for r in tqdm(
                pool.imap_unordered(process_episode_from_item, iterable, chunksize=1),
                total=len(episode_metas),
                desc="Processing episodes",
                mininterval=0.2,
            ):
                results.append(r)
    else:
        results = []
        for episode_id, episode_meta in tqdm(episode_metas.items(), total=len(episode_metas)):
            results.append(process_episode(args, episode_id, episode_meta))

    # Write failed episodes to file
    failed_episodes = [r for r in results if r is not None]
    if failed_episodes:
        with open(os.path.join(os.path.dirname(args.local_base_dir), "failed_episodes.txt"), "w") as f:
            for episode_id, error in failed_episodes:
                f.write(f"Episode {episode_id} failed with error: {error}\n")
        print(f"Number of failed episodes: {len(failed_episodes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_base_dir", type=str, default="datasets/droid_raw_subset/1.0.1_extended",
                        help="The base directory of the replayed rollouts")
    parser.add_argument("--replay_folder", type=str, default="rgb_sim_rollout_masked",
                        help="The folder name of the replayed rollouts")
    parser.add_argument("--droid_raw_dir", type=str, default="/datasets/droid_raw/droid_raw/1.0.1",
                        help="The base directory of the droid raw data")
    parser.add_argument("--path_to_droid_meta", type=str, default="custom/third_party/droid",
                        help="The path to the droid meta data")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use for parallel processing")
    args = parser.parse_args()

    main(args)