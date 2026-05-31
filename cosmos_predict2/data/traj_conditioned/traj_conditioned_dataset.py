
import os
import json
import pickle
import random
import traceback
import warnings
import glob
from tqdm import tqdm
from pathlib import Path
import h5py

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo, uniform_subsample, load_video, VIDEO_RES_SIZE_INFO
from cosmos_predict2.data.traj_conditioned.dataset_utils import convert_euler2quat, convert_to_target_system
from imaginaire.utils import log
from custom.dataset.dataset_meta import DATADIR, CAM_OF_INTEREST_NAME, VIDEO_FOLDER_NAME, LOWDIM_FOLDER_NAME

class TrajConditionedDataset(Dataset):
    def __init__(self, num_frames=93, video_size=(704, 1280), phase='train', specified_task=None, dataset_name="LBM", t5_index="task"):
        super().__init__()
        self.dataset_dir = DATADIR[dataset_name]
        self.sequence_length = num_frames
        self.video_size = video_size
        self.dataset_name = dataset_name

        self.t5_token_max_length = 512
        self.trajectory_max_length = 512

        if specified_task is None: # the data structure should be like: dataset_dir/task_folder
            self.tasks = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d)) and d not in ["t5_xxl", "metas"]]
            self.tasks = sorted(self.tasks)
        else:
            log.warning(f"Using specified task: {specified_task}")
            self.tasks = [specified_task]
        if t5_index == "task":
            self.t5_paths = {}
            for task in self.tasks:
                # TODO: merge pickle files into one
                t5_paths = [os.path.join(self.dataset_dir, task, "t5_xxl", f) for f in os.listdir(os.path.join(self.dataset_dir, task, "t5_xxl")) if f.endswith(".pickle")]
                self.t5_paths[task] = {}
                for t5_path in t5_paths:
                    prompt = os.path.basename(t5_path).replace(".pickle", "").replace("_", " ")
                    self.t5_paths[task][prompt] = t5_path
        elif t5_index == "episode":
            self.t5_paths = {
                episode.stem: episode
                for episode in (Path(self.dataset_dir) / "t5_xxl").glob("*.pickle")
            }

        self.t5 = self._maybe_load_t5_embeddings(self.t5_paths) # {task(or episode_id): {task_description: t5_embedding}(or pickle file path)}

        self.phase = phase
        self._get_episode_paths()

        # log.info(f"list of episode paths for {phase} phase:")
        # log.info("\n".join([episode["path"] for episode in self.episode_paths]))

        log.info(f"{len(self.episode_paths)} episodes in total for {phase} phase -- {self.dataset_name}")

        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def _maybe_load_t5_embeddings(self, t5_paths, threshold=10000):
        """
        we don't expect a large number of t5 embeddings, might be better to load them all at once
        """
        total_t5_files = len(t5_paths)
        if total_t5_files < threshold:
            t5_embeddings = {}
            for task, t5_prompt_path_dict in tqdm(t5_paths.items(), desc="Loading t5 embeddings"):
                t5_embeddings[task] = {}
                if not isinstance(t5_prompt_path_dict, dict):
                    t5_prompt_path_dict = {None: t5_prompt_path_dict}
                for text_prompt, t5_prompt_path in t5_prompt_path_dict.items():
                    with open(t5_prompt_path, "rb") as f:
                        t5_embedding = pickle.load(f)
                        if isinstance(t5_embedding, dict):
                            t5_embeddings[task].update(t5_embedding) # t5_embedding is a dict of {text_prompt: t5_embedding}
                        else:
                            t5_embeddings[task][text_prompt] = t5_embedding # t5_embedding a np.ndarray
            return t5_embeddings
        else:
            return t5_paths

    def _get_episode_list(self, task_dir):
        # try if there are separate episode lists for train and val first,
        # if not, use the all_episode_list_file and naively use the last 1 episode for val
        all_episode_list_file = os.path.join(task_dir, "episode_list.txt")
        episode_list_file = os.path.join(task_dir, f"episode_list_{self.phase}.txt")
        assert os.path.isfile(all_episode_list_file) or os.path.isfile(episode_list_file), f"no episode list file found for task {task_dir}"
        if os.path.isfile(episode_list_file):
            with open(episode_list_file, "r") as f:
                episode_list = f.read().splitlines()
            return episode_list
        else:
            with open(all_episode_list_file, "r") as f:
                episode_list = f.read().splitlines()
            if self.phase == "train":
                return episode_list[:-1]
            else:
                return episode_list[-1:]

    def _get_episode_paths(self):
        episode_paths = []
        for task in self.tasks:
            episode_list = self._get_episode_list(os.path.join(self.dataset_dir, task))
            for episode in episode_list:
                episode_paths.append({
                    "task": task,
                    "path": os.path.join(self.dataset_dir, task, episode),
                    "trajectory_path": os.path.join(self.dataset_dir, task, episode, LOWDIM_FOLDER_NAME[self.dataset_name]),
                })
        self.episode_paths = sorted(episode_paths, key=lambda x: x["path"])

    def __str__(self):
        return f"{len(self.episode_paths)} episodes from {self.dataset_dir}"

    def __len__(self):
        return len(self.episode_paths)

    def _load_video(self, video_path):
        """
        if camera folders are under the video_path, step into each folder, load and concatenate the frames
        """
        if os.path.isfile(os.path.join(video_path, "multiview_video.mp4")):
            return load_video(os.path.join(video_path, "multiview_video.mp4"))
        elif os.path.isfile(video_path):
            return load_video(video_path)
        else:
            # concatenate different cameras into a single video
            cam_folders = [os.path.join(video_path, cam) for cam in CAM_OF_INTEREST_NAME[self.dataset_name]]
            if any(os.path.isdir(cam_folder) for cam_folder in cam_folders):
                frames_data = []
                for cam_folder in cam_folders:
                    frames, fps = load_video(cam_folder, (self.video_size[0] // 2, self.video_size[1] // 2))
                    frames_data.append(frames)
                
                # concatenate the frames
                T, H, W, C = frames_data[0].shape
                if len(frames_data) < 4:
                    # pad with a black video if the number of cameras is less than 4
                    frames_data = frames_data + [np.zeros_like(frames_data[0])] * (4 - len(frames_data))
                top_row = np.concatenate([frames_data[0], frames_data[1]], axis=2)
                bottom_row = np.concatenate([frames_data[2], frames_data[3]], axis=2)
                frames = np.concatenate([top_row, bottom_row], axis=1)
                return frames, fps
            else:
                # if no camera folders, the video_path should be a single view image folder
                return load_video(video_path)

    def _load_videos(self, episode_path):
        """
        load both the condition video and the gt video
        """
        condition_video_path = os.path.join(episode_path, VIDEO_FOLDER_NAME[self.dataset_name]["condition_video"])
        gt_video_path = os.path.join(episode_path, VIDEO_FOLDER_NAME[self.dataset_name]["gt_video"])
        ## handle multiview
        condition_video, _ = self._load_video(condition_video_path)
        gt_video, gt_fps = self._load_video(gt_video_path)

        # there is a small chance that the condition video and the gt video have different lengths,
        # truncate the longer one to the shorter one's length
        video_len = min(len(condition_video), len(gt_video))
        condition_video = condition_video[:video_len]
        gt_video = gt_video[:video_len]
        return condition_video, gt_video, gt_fps, video_len

    def _get_frames(self, episode_path, allow_short_video=False):
        condition_video, gt_video, fps, video_len = self._load_videos(episode_path)
        condition_video = condition_video.astype(np.uint8)
        condition_video = torch.from_numpy(condition_video).permute(0, 3, 1, 2)  # (l, c, h, w)
        gt_video = gt_video.astype(np.uint8)
        gt_video = torch.from_numpy(gt_video).permute(0, 3, 1, 2)  # (l, c, h, w)

        max_start_idx = video_len - self.sequence_length
        if max_start_idx < 0:
            if not allow_short_video:
                return None, None, None, None, None
            else:
                # pad with the last frame id if the video is shorter than the sequence length
                frame_ids = list(range(video_len))
                frame_ids = frame_ids + [frame_ids[-1]] * (self.sequence_length - len(frame_ids))
        else:
            start_frame = np.random.randint(0, max_start_idx+1)
            frame_ids = list(range(start_frame, start_frame + self.sequence_length))

        condition_video = condition_video[frame_ids]
        gt_video = gt_video[frame_ids]
        condition_video = self.preprocess(condition_video)
        gt_video = self.preprocess(gt_video)
        condition_video = torch.clamp(condition_video * 255.0, 0, 255).to(torch.uint8)
        gt_video = torch.clamp(gt_video * 255.0, 0, 255).to(torch.uint8)
        return condition_video, gt_video, fps, frame_ids, video_len

    def _load_trajectory(self, trajectory_path, frame_ids, max_length):
        """
        load the trajectory from the episode path
        """
        if self.dataset_name == "LBM":
            trajectory = []
            timestep_files = sorted(glob.glob(os.path.join(trajectory_path, CAM_OF_INTEREST_NAME[self.dataset_name][0], "*.npz")))
            for timestep_file in timestep_files:
                proprio = np.load(timestep_file, allow_pickle=True)['proprio'].item()
                trajectory.append(
                    np.concatenate([
                        proprio['robot__actual__poses__right::panda__xyz'][0],
                        proprio['robot__actual__poses__right::panda__rot_6d'][0],
                        proprio['robot__actual__poses__left::panda__xyz'][0],
                        proprio['robot__actual__poses__left::panda__rot_6d'][0],
                        proprio['robot__actual__grippers__right::panda_hand'][0],
                        proprio['robot__actual__grippers__left::panda_hand'][0],
                    ], axis=0)
                )
            trajectory = np.stack(trajectory, axis=0)
            task_prompt = None
        elif self.dataset_name == "RoboCasa":
            with open(os.path.join(trajectory_path, "lowdim.pkl"), "rb") as f:
                trajectory = pickle.load(f)
            task_prompt = trajectory['task_description']
            trajectory = np.concatenate([
                trajectory['robot0_base_to_eef_pos'],
                trajectory['robot0_base_to_eef_quat'],
                trajectory['robot0_gripper_qpos'],
            ], axis=1)
        elif self.dataset_name == "piper":
            with open(os.path.join(trajectory_path, "lowdim.pkl"), "rb") as f:
                trajectory = pickle.load(f)
            trajectory = trajectory['sim_replay_end_pose']
            task_prompt = None
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

        trajectory_indicator = np.zeros(trajectory.shape[0]) # indicating which frames are included in this inference
        trajectory_indicator[frame_ids] = 1
        trajectory = np.concatenate([trajectory, trajectory_indicator[:, None]], axis=1)

        if trajectory.shape[0] > max_length:
            log.info(f"Uniformly subsample trajectory from {trajectory.shape[0]} to {max_length}")
            trajectory = uniform_subsample(trajectory, max_length)

        trajectory_padding_mask = torch.ones(max_length, dtype=torch.bool)
        trajectory_padding_mask[:trajectory.shape[0]] = False
        trajectory = torch.from_numpy(trajectory)
        
        trajectory_pad = torch.zeros(max_length, trajectory.shape[1])
        trajectory_pad[:trajectory.shape[0]] = trajectory

        return trajectory_pad, trajectory_padding_mask, task_prompt

    def __getitem__(self, index):
        try:
            data = dict()
            episode_path = self.episode_paths[index]
            condition_video, gt_video, fps, frame_ids, video_len = self._get_frames(episode_path["path"], allow_short_video=True)
            condition_video = rearrange(condition_video, 't c h w -> c t h w')
            gt_video = rearrange(gt_video, 't c h w -> c t h w')
            # Get task folder by finding first directory after self.dataset_dir
            task = episode_path["task"]
            data["video"] = gt_video
            data["control_video"] = condition_video
            data["video_name"] = {
                "video_path": os.path.join(episode_path["path"], VIDEO_FOLDER_NAME[self.dataset_name]["gt_video"]),
                "control_video_path": os.path.join(episode_path["path"], VIDEO_FOLDER_NAME[self.dataset_name]["condition_video"]),
                "frame_ids": frame_ids,
                "total_frames": video_len,
            }

            data["progress_bar"] = torch.tensor(frame_ids)/torch.tensor(video_len) # TODO: might make more sense to divide by video_len - 1 so that the progress bar is 0-1

            trajectory_path = episode_path["trajectory_path"]
            trajectory, trajectory_padding_mask, task_prompt = self._load_trajectory(trajectory_path, frame_ids, self.trajectory_max_length)
            data["trajectory"] = trajectory
            data["trajectory_padding_mask"] = trajectory_padding_mask
            data["trajectory_indicator"] = True

            if task_prompt is not None:
                t5_embedding = self.t5[task][task_prompt]
                if isinstance(t5_embedding, list):
                    if len(t5_embedding) > 1:
                        log.warning(f"Found multiple t5 embeddings for task {task} and prompt {task_prompt}")
                    t5_embedding = t5_embedding[0]
            else:
                t5_embedding = random.choice(list(self.t5[task].values()))

            if isinstance(t5_embedding, str):
                with open(t5_embedding, "rb") as f:
                    t5_embedding = pickle.load(f)[0]  # [n_tokens, 1024]
            n_tokens = t5_embedding.shape[0]
            if n_tokens < self.t5_token_max_length:
                t5_embedding = np.concatenate(
                    [t5_embedding, np.zeros((self.t5_token_max_length - n_tokens, t5_embedding.shape[1]), dtype=np.float32)], axis=0
                )
            t5_text_mask = torch.zeros(self.t5_token_max_length, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1
            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask # NOTE: this is not used so far

            data["fps"] = fps
            data["image_size"] = torch.tensor([self.video_size[0], self.video_size[1], self.video_size[0], self.video_size[1]])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, self.video_size[0], self.video_size[1])

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.episode_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            if self.wrong_number > 3:
                raise ValueError("Too many invalid data encountered")
            return self[np.random.randint(self.__len__())]

class TrajConditionedDataset_droid(TrajConditionedDataset):
    def __init__(self, *args, euler2quat=False, two_view=False, downsample_factor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.euler2quat = euler2quat
        self.two_view = two_view # if True, randomly cut out one of the 3rd views and concatenate the other two views vertically
        self.downsample_factor = downsample_factor

    def _get_episode_paths(self):
        episode_paths = []
        with open(os.path.join(self.dataset_dir, f"episode_meta_{self.phase}.json"), "r") as f:
            episode_metas = json.load(f)
        for episode_id, episode_meta in episode_metas.items():
            episode_paths.append({
                "task": episode_id,
                "path": os.path.join(self.dataset_dir, "1.0.1_extended", episode_meta["rel_path"]),
                "trajectory_path": os.path.join(DATADIR["droid_raw"], episode_meta["rel_path"]),
            })
        self.episode_paths = sorted(episode_paths, key=lambda x: x["path"])

    def _load_videos(self, episode_path):
        condition_video, gt_video, fps, video_len = super()._load_videos(episode_path)
        T_c, H_c, W_c, C_c = condition_video.shape
        T_g, H_g, W_g, C_g = gt_video.shape
        if self.two_view:
            view_ind = np.random.randint(2)
            condition_video = np.concatenate([
                condition_video[:, :H_c//2, view_ind*W_c//2:(view_ind+1)*W_c//2],
                condition_video[:, H_c//2:, :W_c//2],
            ], axis=1)
            gt_video = np.concatenate([
                gt_video[:, :H_g//2, view_ind*W_g//2:(view_ind+1)*W_g//2],
                gt_video[:, H_g//2:, :W_g//2],
            ], axis=1)
        return condition_video, gt_video, fps, video_len
        
    def _load_video(self, video_path):
        video, fps = super()._load_video(video_path)
        T, H, W, C = video.shape
        if self.downsample_factor > 1:
            video = video[::self.downsample_factor]
            fps = fps / self.downsample_factor
        return video, fps

    def _load_trajectory(self, trajectory_path, frame_ids, max_length):
        with h5py.File(os.path.join(trajectory_path, "trajectory.h5"), "r") as f:
            trajectory = np.concatenate([
                f['observation']['robot_state']['cartesian_position'][()],
                f['observation']['robot_state']['gripper_position'][()][:, None],
            ], axis=1)
        if self.euler2quat:
            trajectory = np.concatenate([
                trajectory[:, :3],
                convert_to_target_system(convert_euler2quat(trajectory[:, 3:6])),
                trajectory[:, 6:],
            ], axis=1)
        if self.downsample_factor > 1:
            trajectory = trajectory[::self.downsample_factor]
        task_prompt = None

        trajectory_indicator = np.zeros(trajectory.shape[0]) # indicating which frames are included in this inference
        trajectory_indicator[frame_ids] = 1
        trajectory = np.concatenate([trajectory, trajectory_indicator[:, None]], axis=1)

        if trajectory.shape[0] > max_length:
            log.info(f"Uniformly subsample trajectory from {trajectory.shape[0]} to {max_length}")
            trajectory = uniform_subsample(trajectory, max_length)

        trajectory_padding_mask = torch.ones(max_length, dtype=torch.bool)
        trajectory_padding_mask[:trajectory.shape[0]] = False
        trajectory = torch.from_numpy(trajectory)
        
        trajectory_pad = torch.zeros(max_length, trajectory.shape[1])
        trajectory_pad[:trajectory.shape[0]] = trajectory
        return trajectory_pad, trajectory_padding_mask, task_prompt


class ConcatTrajConditionedDataset(Dataset):
    """
    A dataset that concatenates multiple trajectory conditioned datasets with customizable sampling weights.
    """
    def __init__(self, datasets, weights=None, normalize_weights=True):
        """
        Args:
            datasets (list): List of dataset instances (e.g., TrajConditionedDataset, TrajConditionedDataset_droid, etc.)
            weights (list, optional): List of sampling weights for each dataset. If None, uniform weights are used.
            normalize_weights (bool): Whether to normalize weights to sum to 1.
        """
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.dataset_lengths)
        
        if weights is None:
            weights = [1.0] * len(datasets)
        elif len(weights) != len(datasets):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)})")
        
        if normalize_weights:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        self.weights = weights
        
        # Create cumulative indices for each dataset
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        
        # Create sampling probabilities
        self.sampling_probs = np.array(weights)
        
        log.info(f"ConcatTrajConditionedDataset initialized with {len(datasets)} datasets:")
        for i, (dataset, length, weight) in enumerate(zip(datasets, self.dataset_lengths, weights)):
            log.info(f"  Dataset {i}: {dataset} (length: {length}, weight: {weight:.3f})")
        log.info(f"Total length: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        """
        Sample from datasets according to their weights, then sample an item from the selected dataset.
        """
        # Sample a dataset according to weights
        dataset_idx = np.random.choice(len(self.datasets), p=self.sampling_probs)
        
        # Sample a random index from the selected dataset
        selected_dataset = self.datasets[dataset_idx]
        item_idx = np.random.randint(len(selected_dataset))
        
        # Get the item and add dataset info
        data = selected_dataset[item_idx]
        
        # Add metadata about which dataset this sample came from
        data["dataset_index"] = dataset_idx
        data["dataset_name"] = getattr(selected_dataset, 'dataset_name', f'dataset_{dataset_idx}')
        
        return data

    def __str__(self):
        dataset_info = []
        for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            dataset_info.append(f"Dataset {i} (weight: {weight:.3f}): {dataset}")
        return f"ConcatTrajConditionedDataset with {len(self.datasets)} datasets:\n" + "\n".join(dataset_info)


if __name__ == "__main__":
    # dataset = TrajConditionedDataset(dataset_name="RoboCasa", num_frames=189, video_size=(256, 256), phase='train', specified_task=None)
    # dataset = TrajConditionedDataset(dataset_name="piper", num_frames=93, video_size=(360, 320), phase='train', specified_task=None)
    # dataset = TrajConditionedDataset(dataset_name="LBM", num_frames=93, video_size=(256, 256), phase='train', specified_task="TurnMugRightsideUp")
    dataset = TrajConditionedDataset_droid(dataset_name="droid", num_frames=93, video_size=(368, 368), phase='val', t5_index="episode",
        two_view=True, euler2quat=True, downsample_factor=3
    )

    dataset = ConcatTrajConditionedDataset(
        datasets=[
            TrajConditionedDataset(dataset_name="piper", num_frames=93, video_size=(368, 368), phase='train', specified_task=None),
            TrajConditionedDataset_droid(dataset_name="droid", num_frames=93, video_size=(368, 368), phase='val', t5_index="episode", two_view=True, euler2quat=True, downsample_factor=3),
        ],
        weights=[0.6, 0.4],
        normalize_weights=True,
    )

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        breakpoint()