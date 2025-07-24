import subprocess
import os

def get_video_duration(file_path):
    """使用ffprobe获取视频持续时间（秒）。"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    duration = float(result.stdout)
    return duration

        
def sample_video(input_file, output_folder):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    duration = get_video_duration(input_file)
    
    segment_duration = 3  # 目标片段长度（秒）
    num_segments = int(duration / segment_duration)
    remainder = duration % segment_duration
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_segments):
        start_time = i * segment_duration
        output_file = os.path.join(output_folder, f"{base_name}_sample_{i+1}.mp4")
        command = [
            'ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(segment_duration),
            '-avoid_negative_ts', 'make_zero', '-c:v', 'libx264', '-c:a', 'aac', output_file
        ]
        subprocess.run(command)
    
    # 处理视频结尾不足3秒的情况
    if remainder > 0:
        pass
        #start_time = num_segments * segment_duration
        #output_file = os.path.join(output_folder, f"{base_name}_sample_{num_segments+1}.mp4")
        #command = [
        #    'ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(remainder),
        #    '-avoid_negative_ts', 'make_zero', '-c:v', 'libx264', '-c:a', 'aac', output_file
        #]
        #subprocess.run(command)


def process_videos(input_folder, output_folder):
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    for video_file in video_files:
        sample_video(video_file, output_folder)

def main():
    input_folder = '../screenRecordingsall'
    output_folder = '../video_clips'  
    process_videos(input_folder, output_folder)

if __name__ == "__main__":
    main()
