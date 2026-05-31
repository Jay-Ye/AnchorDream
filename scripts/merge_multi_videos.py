import os
import numpy as np
import glob
import mediapy as mp

video_path = "."

# Get all video files in the current directory
video_files = glob.glob(os.path.join(video_path, "video_0*"))
video_files.sort()  # Sort to ensure consistent ordering

if len(video_files) != 8:
    print(f"Found {len(video_files)} videos, but expected 8")
    print("Video files found:", video_files)

# Read all videos and get their properties
videos = []
frame_counts = []
fps_list = []
widths = []
heights = []
for video_file in video_files:
    video = mp.read_video(video_file)
    videos.append(video)
    frame_counts.append(len(video))
    # MediaPy doesn't directly provide fps, so we'll use a default or extract from metadata
    fps_list.append(30)  # Default fps, adjust as needed
    heights.append(video.shape[1])
    widths.append(video.shape[2])

# Use the minimum frame count and first video's fps
min_frames = min(frame_counts)
output_fps = fps_list[0]
output_width = max(widths) * 2  # Width for 2 columns
output_height = max(heights) * 4  # Height for 4 rows (8 videos in 2x4 grid)

print(f"Merging {len(video_files)} videos in 2 columns...")
print(f"Output dimensions: {output_width}x{output_height}")
print(f"Total frames: {min_frames}")

# Process each frame
merged_frames = []
for frame_idx in range(min_frames):
    frames = []
    
    # Get frame from each video
    for video in videos:
        frame = video[frame_idx]
        # Resize frame to standard size if needed
        if frame.shape[1] != max(widths) or frame.shape[0] != max(heights):
            # MediaPy uses HWC format, so we need to handle resizing differently
            import cv2  # Still need cv2 for resize
            frame = cv2.resize(frame, (max(widths), max(heights)))
        frames.append(frame)
    
    # Arrange frames in 2 columns, 4 rows
    rows = []
    for row in range(4):
        left_frame = frames[row * 2]
        right_frame = frames[row * 2 + 1]
        row_frame = np.hstack([left_frame, right_frame])
        rows.append(row_frame)
    
    # Vertically concatenate all rows
    merged_frame = np.vstack(rows)
    merged_frames.append(merged_frame)
    
    if frame_idx % 100 == 0:
        print(f"Processed {frame_idx}/{min_frames} frames")

# Convert to numpy array and save
merged_video = np.array(merged_frames)
mp.write_video('merged_video_new.mp4', merged_video, fps=output_fps)

print("Video merging completed! Output saved as 'merged_video_new.mp4'")