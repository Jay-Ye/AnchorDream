import os
data_dir = "datasets/lbm_data_demo"
task = "BimanualStackPlatesOnTableFromDryingRack"
episode_folder = os.path.join(data_dir, task)
stations = os.listdir(episode_folder)
episode_list = []
for station in stations:
    if station == "metas" or station == "t5_xxl":
        continue
    real_sim_path = os.listdir(os.path.join(episode_folder, station))
    for real_sim in real_sim_path:
        temp_episodes = os.listdir(os.path.join(episode_folder, station, real_sim))
        for episode in temp_episodes:
            temp_episode_path = os.path.join(episode_folder, station, real_sim, episode)
            if os.path.exists(os.path.join(temp_episode_path, "rgb")) and os.path.exists(os.path.join(temp_episode_path, "rgb_sim_rollout_masked")) and os.path.exists(os.path.join(temp_episode_path, "lowdim")):
                episode_list.append(temp_episode_path)

with open(os.path.join(episode_folder, "episode_list.txt"), "w") as f:
    for episode in episode_list:
        f.write(episode + "\n")