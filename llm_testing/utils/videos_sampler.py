import subprocess
import os
from multiprocessing import Pool, cpu_count

video_processed_list = "video_processed.txt"

def get_video_duration(file_path):
    """Use ffprobe to acquire the video duration(sec)."""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout)
    return duration

def clean_partial_files(output_folder, base_name, n):
    """Delete Unfinished CLips"""
    for i in range(n):
        partial_file = os.path.join(output_folder, f"sample_{i+1}_{base_name}")
        if os.path.exists(partial_file):
            os.remove(partial_file)
            print(f"Deleted partial file: {partial_file}")

def ffmpeg_sample_video(input_file, output_folder, n, m, processed_videos):
    base_name = os.path.basename(input_file)
    
    if base_name in processed_videos:
        print(f"Skipping {base_name}, already processed.")
        return

    clean_partial_files(output_folder, base_name, n)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    duration = get_video_duration(input_file)  # Acquire the actual duration of video
    # Ignore all content before start point 
    start_point = 15
    intervals = (duration - start_point) / n
    
    success = True
    for i in range(n):
        start_time = start_point + i * intervals
        output_file = os.path.join(output_folder, f"sample_{i+1}_{base_name}")
        command = [
            'ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(m), 
            '-c:v', 'libx264', '-c:a', 'aac', output_file
        ]
        result = subprocess.run(command)
        if result.returncode != 0:
            success = False
            break
    
    if success:
        with open(video_processed_list, "a") as file:
            file.write(base_name + "\n")
    else:
        clean_partial_files(output_folder, base_name, n)

def process_videos(input_folder, output_folder, n, m):
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    if os.path.exists(video_processed_list):
        with open(video_processed_list, "r") as file:
            processed_videos = {line.strip() for line in file}
    else:
        processed_videos = set()
    
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(ffmpeg_sample_video, [(file, output_folder, n, m, processed_videos) for file in video_files])

def main():
    input_folder = '../../screenRecordingsall'
    output_folder = 'videos'
    n = 3  # 片段数量
    m = 3  # 每个片段的秒数

    process_videos(input_folder, output_folder, n, m)

if __name__ == "__main__":
    main()
