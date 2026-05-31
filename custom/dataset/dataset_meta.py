DATADIR = {
    "RoboCasa": "datasets/RoboCasa",
    "droid": "datasets/droid_raw_subset",
    "droid_raw": "datasets/droid_raw",
    "piper": "datasets/piper_data_cosmos"
}
CAM_OF_INTEREST_NAME = {
    "RoboCasa": ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"],
    "droid": ["external_cam1", "external_cam2", "wrist_cam"],
    "piper": ["top_camera", "right_camera"]
}
VIDEO_FOLDER_NAME = {
    "RoboCasa": {
        "condition_video": "rgb_sim_rollout_masked",
        "gt_video": "rgb",
    },
    "droid": {
        "condition_video": "input_multiview_video.mp4",
        "gt_video": "gt_multiview_video.mp4",
    },
    "piper": {
        "condition_video": "rgb_sim_rollout_masked",
        "gt_video": "rgb",
    },
}
LOWDIM_FOLDER_NAME = {
    "RoboCasa": "lowdim",
    "droid": None,
    "piper": "lowdim",
}

