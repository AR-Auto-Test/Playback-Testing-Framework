# frames_sampler.py
import cv2
import numpy as np
import os
from scipy.io import savemat

def sample_frames_from_video(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count-1, n_frames, dtype=int)
    
    for i in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
    cap.release()
    return np.array(frames)

def save_frames_to_mat(video_dir, output_file, n_frames):
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_data = [sample_frames_from_video(video, n_frames) for video in video_files]
    savemat(output_file, {'frames': video_data})

if __name__ == '__main__':
    video_dir = 'data/videos'
    output_file = 'data/video_frames.mat'
    n_frames = 5  # Example frame count, adjust as needed
    save_frames_to_mat(video_dir, output_file, n_frames)
    print("Frame sampling done.")