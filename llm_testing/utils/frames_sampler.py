# frames_sampler.py
import cv2
import numpy as np
import random
import os
import torch
from scipy.io import savemat
import h5py
import re
from datetime import datetime, timedelta

# Metadata Processing

def parse_metadata(metadata_file):
    metadata = {}
    
    # 读取metadata文件
    with open(metadata_file, "r") as file:
        metadata_content = file.read()
        
    # 使用正则表达式根据空白行进行分割
    metadata_blocks = re.split(r'\n\s*\n', metadata_content)
    
    # 遍历每个视频块
    for block in metadata_blocks:
        lines = block.strip().split("\n")
        if len(lines) > 0:
            video_name = lines[0].rstrip(":")
            metadata[video_name] = lines[1:]
    
    return metadata

def get_clip_start_time(video_name, global_start_time):
    match = re.search(r'_sample_(\d+)\.mp4', video_name)
    if match:
        n = int(match.group(1))
        return global_start_time + timedelta(seconds=3 * (n - 1))
    return None

def get_nearest_metadata(metadata, timestamp):
    nearest_data = {}
    for key in ['XiaoyiYang_Position', 'XiaoyiYang_Rotation', 'XiaoyiYang_Camera_Position', 'XiaoyiYang_Camera_Rotation']:
        min_diff = timedelta(days=365)
        nearest_line = None
        for line in metadata:
            if key in line:
                data_time = datetime.strptime(line[:18], '%m-%d %H:%M:%S.%f')
                diff = abs(data_time - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    nearest_line = line
        #print(nearest_line)
        if nearest_line:
            value = parse_metadata_line(nearest_line)
            nearest_data[key] = value
        else:
            raise Exception(f"Missing metadata for key: {key} at timestamp: {timestamp}")
        #print(nearest_data)
    return nearest_data

def parse_metadata_line(line):
    parts = line.split(':')
    try:
        value = [float(x) for x in parts[-1].strip()[1:-1].split(',')]
    except Exception as e:
        print(e)
    
    return value
        
    

def calculate_distance(position, camera_position):
    dx = position[0] - camera_position[0]
    dy = position[1] - camera_position[1]
    dz = position[2] - camera_position[2]
    return (dx**2 + dy**2 + dz**2)**0.5



# Frame data processing


def crop_frame(frame, crop_size):
    """
    Crop the center region of the frame to the given size.

    Args:
        frame: The frame to be cropped (as a numpy array).
        crop_size: A tuple of (height, width) indicating the size of the crop.

    Returns:
        The cropped frame.
    """
    h, w = frame.shape[1:3]  # Assuming frame is in (C, H, W) format
    start_x = w // 2 - crop_size[1] // 2
    start_y = h // 2 - crop_size[0] // 2
    return frame[:, start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]


def random_crop(frame, crop_size):
    """
    Perform a random crop on the frame.

    Args:
        frame: The frame to be cropped, assumed to be in (C, H, W) format.
        crop_size: A tuple of (crop_height, crop_width) indicating the size of the crop.

    Returns:
        A randomly cropped section of the original frame.
    """
    C, H, W = frame.shape
    crop_height, crop_width = crop_size

    # Ensure the crop size is not larger than the frame size
    if crop_height > H or crop_width > W:
        raise ValueError("Crop size must be smaller than the frame size.")

    # Randomly choose the top-left corner of the cropping area
    start_y = np.random.randint(0, H - crop_height + 1)
    start_x = np.random.randint(0, W - crop_width + 1)

    # Crop and return the frame
    return frame[:, start_y:start_y + crop_height, start_x:start_x + crop_width]


def supplement_frame_indices(frame_idxs, n_frames):
    """
    Supplement the list of frame indices by inserting duplicates until reaching the desired count.
    
    Args:
        frame_idxs: List of currently selected frame indices.
        n_frames: The desired total number of frames to be sampled.
        
    Returns:
        A list of frame indices supplemented with duplicates to meet the required number of frames.
    """
    while len(frame_idxs) < n_frames:
        index_to_duplicate = random.randint(0, len(frame_idxs) - 1)
        if random.random() > 0.5:
            if index_to_duplicate + 1 < len(frame_idxs):
                frame_idxs.insert(index_to_duplicate + 1, frame_idxs[index_to_duplicate])
            else:
                frame_idxs.append(frame_idxs[index_to_duplicate])
        else:
            frame_idxs.insert(index_to_duplicate, frame_idxs[index_to_duplicate])
    return frame_idxs

# Core Methods
def sample_frames_from_video(video_path, metadata, n_frames=20, sample='uniform', black_threshold=0.98):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    acc_samples = min(n_frames, frame_count)
    sample_indices = np.linspace(0, frame_count, acc_samples + 1, dtype=int)
    
    ranges = []
    for i, interval in enumerate(sample_indices[:-1]):
        ranges.append((interval, sample_indices[i + 1] - 1))

    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    else:  # sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        
    video_name = os.path.basename(video_path)
    global_start_time = None
    for line in metadata[video_name]:
        if 'XiaoyiYang_playback_start' in line:
            global_start_time = datetime.strptime(line[:18], '%m-%d %H:%M:%S.%f')
            break
    
    if global_start_time is None:
        raise ValueError(f"No XiaoyiYang_playback_start found for video {video_name}")
    
    clip_start_time = get_clip_start_time(video_name, global_start_time)
    
    frames = []
    all_metadata = []
    last_non_black_frame = None
    for i in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            n_tries = 5
            for _ in range(n_tries):
                ret, frame = cap.read()
                if ret:
                    break
            if not ret:
                if last_non_black_frame is not None:
                    frame = last_non_black_frame
                else:
                    raise ValueError("Error reading frame and no non-black frame available.")
                        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            #frame = random_crop(frame, crop_size=(1080, 1920))
            frames.append(frame)
            
            frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            metadata_timestamp = clip_start_time + timedelta(seconds=frame_timestamp)
            nearest_metadata = get_nearest_metadata(metadata[video_name], metadata_timestamp)
            
            #print(nearest_metadata.values())
            
            position = nearest_metadata['XiaoyiYang_Position']
            rotation = nearest_metadata['XiaoyiYang_Rotation']
            camera_position = nearest_metadata['XiaoyiYang_Camera_Position']
            camera_rotation = nearest_metadata['XiaoyiYang_Camera_Rotation']
            
            
            #print(position)
            #print(rotation)
            #print(camera_position)
            #print(camera_rotation)
            distance = calculate_distance(position, camera_position)
            
            frame_metadata = position + rotation + camera_position + camera_rotation + [distance]
            all_metadata.append(frame_metadata)
            
        else:
            raise ValueError
        
    cap.release() 
    
    return np.array(frames), np.array(all_metadata)

def save_frames_to_hdf5(video_dir, metadata_file, output_file, frame_num):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    metadata = parse_metadata(metadata_file)
    
    processed_videos = set()
    if os.path.exists(output_file):
        with h5py.File(output_file, 'r') as hf:
            processed_videos = set(hf.keys())
        print(f"Found {len(processed_videos)} processed videos in the output file.")
    
    count = 0
    with h5py.File(output_file, 'a') as hf:
        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            if video_name in processed_videos:
                print(f"Skipping {video_name} as it is already processed.")
                count += 1
                continue
            
            print(f"Processing {video_name} ({count}/{len(video_files)})")
            count += 1
            video_path = os.path.join(video_dir, video_file)
            
            try:
                frames, all_metadata = sample_frames_from_video(video_path, metadata, frame_num)
            except ValueError as e:
                print(f"Error processing {video_name}: {str(e)}")
                continue
            
            hf.create_dataset(f'{video_name}/frames', data=frames)
            hf.create_dataset(f'{video_name}/metadata', data=all_metadata)
            
            
if __name__ == '__main__':
    video_dir = 'video_clips'
    metadata_file = 'metadata_clips_formatUnified_removeNull.txt'
    frame_num = 32
    output_file = f"video_metadata_{frame_num}.hdf5"
    save_frames_to_hdf5(video_dir, metadata_file, output_file, frame_num)
    print("Frame sampling and metadata extraction done.")