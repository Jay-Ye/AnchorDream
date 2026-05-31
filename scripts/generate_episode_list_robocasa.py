import os
data_dir = "datasets/RoboCasa"
task = "OpenDrawer"
episode_folder = f"{data_dir}/v0.1/single_stage/kitchen_drawer/{task}/2024-05-03/demo_gentex_im128_randcams_im256/"
episode_list = os.listdir(episode_folder)

with open(os.path.join(data_dir, f"{task}_episode_list.txt"), "w") as f:
    for episode in episode_list:
        f.write(os.path.join(episode_folder, episode) + "\n")