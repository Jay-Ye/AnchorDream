import os
import imageio
from PIL import Image, ImageDraw, ImageFont
import glob
from tqdm import tqdm
import numpy as np
import math

def video_to_gif(video_path, output_path, fps=10):
    """Convert video to gif and save it
    Args:
        video_path (str): Path to input video
        output_path (str): Path to save output gif
        fps (int): Frames per second for output gif
    """
    # Read video
    reader = imageio.get_reader(video_path)
    frames = []
    # Only keep every 3rd frame to reduce size
    for i, frame in enumerate(reader):
        if i % 5 == 0:  # Skip 4 out of every 5 frames
            frames.append(Image.fromarray(frame))
    reader.close()
    
    # Save as gif
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//fps,
        loop=0
    )

def videos_to_grid_gif(video_paths, output_path, num_frames=50, fps=10, grid_size=None):
    """Convert multiple videos to a single gif arranged in a grid
    Args:
        video_paths (list): List of paths to input videos
        output_path (str): Path to save output gif
        num_frames (int): Number of frames to sample from each video
        fps (int): Frames per second for output gif
        grid_size (tuple): Grid size as (rows, cols). If None, will be automatically determined
    """
    # Read all videos and sample frames
    all_frames = []
    for video_path in video_paths:
        reader = imageio.get_reader(video_path)
        video_frames = []
        # Only keep every 3rd frame to reduce size
        for i, frame in enumerate(reader):
            if i % 5 == 0:  # Skip 4 out of every 5 frames
                video_frames.append(frame)
        reader.close()
        
        # Uniform sampling
        if len(video_frames) > num_frames:
            indices = np.linspace(0, len(video_frames)-1, num_frames, dtype=int)
            video_frames = [video_frames[i] for i in indices]
        all_frames.append(video_frames)
    
    # Make all videos same length by repeating last frame
    max_frames = max(len(frames) for frames in all_frames)
    for frames in all_frames:
        while len(frames) < max_frames:
            frames.append(frames[-1])
    
    # Determine grid size if not provided
    if grid_size is None:
        n = len(video_paths)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        grid_size = (rows, cols)
    
    # Create grid frames
    grid_frames = []
    for frame_idx in range(max_frames):
        # Get all frames at this time step
        current_frames = [frames[frame_idx] for frames in all_frames]
        
        # Create grid
        rows, cols = grid_size
        cell_height = max(frame.shape[0] for frame in current_frames)
        cell_width = max(frame.shape[1] for frame in current_frames)
        
        grid = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)
        
        # Convert to PIL Image for drawing text
        grid_img = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_img)
        
        for idx, frame in enumerate(current_frames):
            i, j = idx // cols, idx % cols
            h, w = frame.shape[:2]
            y, x = i * cell_height, j * cell_width
            
            # Add frame to grid
            grid_img_array = np.array(grid_img)
            grid_img_array[y:y+h, x:x+w] = frame
            grid_img = Image.fromarray(grid_img_array)
            draw = ImageDraw.Draw(grid_img)
            
            # Add folder name as text
            folder_name = os.path.basename(os.path.dirname(video_paths[idx]))
            text_x = x + 10
            text_y = y + 10
            draw.text((text_x, text_y), folder_name, fill=(255, 255, 255))
            
        grid_frames.append(grid_img)
    
    # Save as gif
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_frames[0].save(
        output_path,
        save_all=True,
        append_images=grid_frames[1:],
        duration=1000//fps,
        loop=0
    )
    
    # Also save as mp4
    mp4_path = output_path.replace('.gif', '.mp4')
    frames_array = [np.array(frame) for frame in grid_frames]
    imageio.mimsave(mp4_path, frames_array, fps=fps)

# Process all kitchen folders
base_dir = "results/cosmos_nemo_assets/robocasa_189"
save_dir = "results/cosmos_nemo_assets/robocasa_189_gif"
kitchen_folders = [f for f in os.listdir(base_dir) if f.startswith("kitchen")]

video_paths = []
for folder in tqdm(kitchen_folders, desc="Processing folders"):
    folder_path = os.path.join(base_dir, folder)
    
    # Find first video file
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    if not video_files:
        print(f"No video files found in {folder_path}")
        continue
        
    video_path = video_files[0]
    video_paths.append(video_path)
    gif_path = os.path.join(save_dir, f"{folder}.gif")
    
    # # Convert to gif
    # video_to_gif(video_path, gif_path)
    # print(f"Converted {video_path} to {gif_path}")

videos_to_grid_gif(video_paths, f"{save_dir}/grid_30.gif", num_frames=30, grid_size=(6,4))