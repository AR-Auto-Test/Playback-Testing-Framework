from moviepy.editor import VideoFileClip, ImageSequenceClip
import os

def stretch_video_by_frame_repetition(input_path, output_path, target_duration=10):
    # 如果目标文件已存在，跳过处理
    if os.path.exists(output_path):
        print(f"跳过: {output_path} (已存在)")
        return
    
    print(f"处理: {input_path} -> {output_path}")
    
    # 加载视频
    clip = VideoFileClip(input_path)
    
    # 获取原始持续时间和总帧数
    original_duration = clip.duration
    total_frames = clip.reader.nframes
    
    # 计算每帧需要重复的次数
    #frame_repetition_count = int(target_duration / original_duration * clip.fps)
    #frame_repetition_count = int(target_duration / original_duration)
    frame_repetition_count = 3 # Or simply set how many times
    
    # 提取帧并重复
    frames = []
    for frame in clip.iter_frames():
        frames.extend([frame] * frame_repetition_count)  # 将每一帧重复指定次数
    
    # 创建新的视频剪辑
    new_clip = ImageSequenceClip(frames, fps=clip.fps)
    
    # 保存新视频
    new_clip.write_videofile(output_path, codec="libx264")

# 示例批量处理
def batch_process_videos(input_folder, output_folder, target_duration=10):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有视频文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):  # 只处理 mp4 格式文件
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 调用拉伸函数
            stretch_video_by_frame_repetition(input_path, output_path, target_duration)

# 使用示例
batch_process_videos("exp/optimize_prompt/few_shot", "exp/optimize_prompt/few_shot_gemini", target_duration=15)
