"""
python 5_label_claude_thinking.py --mode discard (to start fresh)
python 5_label_claude_thinking.py --mode continue (to resume from where it left off)
python 5_label_claude_thinking.py --video_dir path/to/videos --output_dir output_folder

This script focuses on the third round of conversations, retrieving round1 and round2 results
from existing JSON files in the 5_output directory.

此程序为非batch单进程版本，cost较高，主要用于当5_label_claude_batch_thinking出现timeout或其他错误时少量数据的生成。
round1和round2的部分建议依然使用batch版本生成，此程序专用于round3部分
"""

import cv2
import base64
import time
import os
import json
import re
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
import traceback

load_dotenv()  # Load environment variables from .env file

class ARVideoProcessor:
    MODEL = "claude-3-7-sonnet-20250219"  # Using Claude 3.7 Sonnet
    FRAME_INTERVAL = 25  # Frame sampling interval
    MAX_TOKENS = 20000
    MAX_RETRIES = 1  # maximum number of retry attempts
    TEMP_DIR = "temp_frames"  # Directory to cache frames
    
    METRICS = [
        "Object Placement",
        "Occlusion",
        "Object Movement",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    ]

    def __init__(self, output_dir: str = "5_output"):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=120.0)
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.video_mapping = {}  # 视频名称到编号的映射
        self.reverse_mapping = {}  # 编号到视频名称的映射
        # Create temp directory for frames
        self.temp_frames_dir = Path(self.TEMP_DIR)
        self.temp_frames_dir.mkdir(exist_ok=True)
        
    def setup_directories(self):
        """Create necessary output directories if they don't exist."""
        directories = ['logs', 'json', 'mapping', self.TEMP_DIR]
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def initialize_video_mapping(self, video_dir: str):
        """为所有视频创建编号映射."""
        video_files = sorted([f for f in os.listdir(video_dir) 
                            if f.endswith(('.mp4', '.mov'))])
        
        mapping_file = self.output_dir / "mapping" / "video_mapping.json"
        
        # 如果映射文件已存在，则加载它
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    self.video_mapping = mapping_data.get("video_to_id", {})
                    self.reverse_mapping = mapping_data.get("id_to_video", {})
                print(f"Loaded existing mapping for {len(self.video_mapping)} videos")
                return self.video_mapping
            except Exception as e:
                print(f"Error loading existing mapping: {e}, creating new mapping")
        
        # 创建新的映射
        for idx, video_name in enumerate(video_files, 1):
            video_id = f"v{idx:04d}"  # 使用4位数字，前导零填充
            self.video_mapping[video_name] = video_id
            self.reverse_mapping[video_id] = video_name
        
        # 保存映射到文件
        with open(mapping_file, 'w') as f:
            json.dump({
                "video_to_id": self.video_mapping,
                "id_to_video": self.reverse_mapping
            }, f, indent=4)
            
        print(f"Created mapping for {len(video_files)} videos")
        return self.video_mapping

    def get_video_id(self, video_name: str) -> str:
        """获取视频的编号ID."""
        return self.video_mapping.get(video_name, "unknown")

    def get_video_name(self, video_id: str) -> str:
        """根据编号ID获取原始视频名称."""
        return self.reverse_mapping.get(video_id, "unknown")
    
    def get_frames_path(self, video_name: str) -> str:
        """Get the path to the cached frames file for a video."""
        video_id = self.get_video_id(video_name)
        return str(self.temp_frames_dir / f"{video_id}_frames.pkl")
    
    def save_frames(self, video_name: str, frames: list) -> None:
        """Save video frames to a temporary file."""
        frames_path = self.get_frames_path(video_name)
        with open(frames_path, 'wb') as f:
            pickle.dump(frames, f)
            
    def load_frames(self, video_name: str) -> list:
        """Load video frames from temporary file if available."""
        frames_path = self.get_frames_path(video_name)
        if os.path.exists(frames_path):
            with open(frames_path, 'rb') as f:
                return pickle.load(f)
        return None
            
    def resize_frame(self, frame, max_size=2000):
        """Resize frame while maintaining aspect ratio if it exceeds max size."""
        height, width = frame.shape[:2]

        # Check if resizing is needed
        if max(height, width) <= max_size:
            return frame

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def read_video_frames(self, video_path: str) -> list:
        """Read and sample video frames, converting to base64."""
        video = cv2.VideoCapture(video_path)
        base64Frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Resize frame if necessary
            resized_frame = self.resize_frame(frame)

            # Convert to JPEG and then to base64
            _, buffer = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        # Sample frames at regular intervals
        sampled_frames = base64Frames[0::self.FRAME_INTERVAL]
        print(f"{len(base64Frames)} frames read from {video_path}, {len(sampled_frames)} frames sampled")

        return sampled_frames
    
    def generate_system_prompt(self) -> str:
        """Generate the system prompt for Claude."""
        return """You are a specialized AI assistant trained to review AR applications based on screen recordings. The video provided is a screen recording from a mobile phone that demonstrates an AR (Augmented Reality) application. We specifically designed and selected certain environment and use cases to test the AR application to identify any potential performance issues with AR functionality. In all the screen recording videos, the movement of AR objects is simulated programmatically as interactions on the phone screen. Currently, the simulated interactions include the following two types: 1.Tapping the screen: The AR object will immediately appear at the corresponding position. 2. Swiping the screen: The AR object will move along the swipe path to the corresponding positions. 
    
        Your task is to evaluate the realism of AR effect and the performance of AR apps based on following metrics: 1. Object Placement. 2. Object Movement. 3. Occlusion. 4. Lighting. 5. Visual Artifacts and Rendering Issues. 6. Black Screen 

        And you could ignore the following aspects in evaluation: 
        1. The size of AR object. Because some apps may privide the resize feature so that the user could adjust. 
        2. The stylish or cartoonish AR object. Because some apps may provide stylish and cartoonish models according to the theme of app. The The stylish or cartoonish does not account for realism issue. 
        3. The UI elements. The effect and visibility of UI elements like grid lines or measures are part of design by developers. Therefore, the interaction between AR object and UI element is out of the scope of evaluation.

        For evaluation questions, I will ask you to response in JSON format. Please ensure your response is a valid JSON object with this format:
        {
            "Video_name": string,   
            "Metrics": string, // the metric to evaluate, e.g. Object Placement
            "Issue": boolean, // true if there are issues found 
            "Reason":  string //The reason why you think there are issues or no issues. Provide explanation.
        }
        """
    
    def get_processed_videos(self) -> dict:
        """获取已处理的视频及其已处理的指标"""
        processed = self.get_processed_round3_items()
        return processed
    
    def get_processed_round3_items(self) -> Dict[str, Set[str]]:
        """Load already processed videos and their metrics from label_claude.json."""
        json_file = self.output_dir / "json" / "label_claude.json"
        processed = {}  # Dict[video_id, Set[metric]]

        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    content = f.read()

                # 分割并解析多个JSON对象
                json_objects = []
                current = ""
                depth = 0

                for char in content:
                    if char == '{':
                        if depth == 0:
                            current = ""
                        depth += 1
                    elif char == '}':
                        depth -= 1
                    current += char

                    if depth == 0 and current.strip():
                        try:
                            result = json.loads(current)
                            # 检查必要的字段是否存在
                            if 'custom_id' in result and 'Metrics' in result:
                                video_id = result['custom_id'].split('_')[0]
                                metric = result['Metrics']

                                if video_id not in processed:
                                    processed[video_id] = set()
                                processed[video_id].add(metric)

                            current = ""
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse JSON object: {current[:100]}...")

                #if processed:
                #    print(f"Found {len(processed)} processed videos in label_claude.json")
                #    for video_id, metrics in processed.items():
                #        print(f"  {video_id}: {len(metrics)} metrics processed")
                #else:
                #    print("No processed items found in label_claude.json")

            except Exception as e:
                print(f"Error reading label_claude.json: {str(e)}")

        return processed


    def load_round_results(self, round_num: int) -> dict:
        """Load results from a specific round with better error handling."""
        result_file = self.output_dir / f"round{round_num}" / "results.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading results from round {round_num}: {str(e)}")
                print("Attempting to fix corrupted JSON file...")
                
                # Try to fix corrupted JSON by merging multiple JSON objects
                try:
                    with open(result_file, 'r') as f:
                        content = f.read()
                    
                    # Split content into separate JSON objects and parse each
                    json_objects = []
                    current = ""
                    depth = 0
                    
                    for char in content:
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            
                        current += char
                        
                        if depth == 0 and current.strip():
                            try:
                                obj = json.loads(current)
                                json_objects.append(obj)
                                current = ""
                            except json.JSONDecodeError:
                                pass
                    
                    # Merge all valid JSON objects
                    merged_results = {}
                    for obj in json_objects:
                        merged_results.update(obj)
                    
                    # Save fixed results
                    with open(result_file, 'w') as f:
                        json.dump(merged_results, f, indent=4)
                    
                    return merged_results
                    
                except Exception as fix_error:
                    print(f"Could not fix JSON file: {str(fix_error)}")
                    return {}
        else:
            print(f"Results file for round {round_num} not found: {result_file}")
        return {}

    def process_video(self, video_path: str) -> None:
        """Process a single video with Claude and logs both complete responses and JSON data."""
        try:
            video_name = os.path.basename(video_path)
            video_id = self.get_video_id(video_name)

            print(f"Processing video: {video_name} (ID: {video_id})")

            # 获取已处理的视频和指标
            processed_videos = self.get_processed_videos()

            # 检查此视频的哪些指标已经处理过
            processed_metrics = processed_videos.get(video_id, set())
            remaining_metrics = [m for m in self.METRICS if m not in processed_metrics]

            if not remaining_metrics:
                print(f"Skipping {video_id} - all metrics already processed")
                return
            else:
                if processed_metrics:
                    print(f"{video_id} has {len(processed_metrics)}/{len(self.METRICS)} metrics processed")
                    print(f"Remaining metrics to process: {', '.join(remaining_metrics)}")
                else:
                    print(f"{video_id} has no metrics processed yet")

            # 加载前两轮的结果
            round1_results = self.load_round_results(1)
            round2_results = self.load_round_results(2)

            # 验证是否有前两轮的结果
            if video_id not in round1_results:
                print(f"Warning: No round 1 results found for {video_id} (ID: {video_id}). Continuing anyway...")
            else:
                print(f"Found round 1 results for {video_id}")

            if video_id not in round2_results:
                print(f"Warning: No round 2 results found for {video_id} (ID: {video_id}). Continuing anyway...")
            else:
                print(f"Found round 2 results for {video_id}")

            # 尝试从缓存加载帧
            frames = self.load_frames(video_name)
            if frames is None:
                print(f"No cached frames found for {video_name}, reading video...")
                frames = self.read_video_frames(video_path)
                self.save_frames(video_name, frames)
                print(f"Frames saved for {video_name}")
            else:
                print(f"Loaded {len(frames)} cached frames for {video_name}")

            # 准备输出文件
            output_file = self.output_dir / "logs" / f"{video_id}_conversation.txt"
            json_output_file = self.output_dir / "json" / "label_claude_thinking.json"

            # 初始化结果日志
            result_log = [f"Processing video: {video_path}\n\n"]

            # 添加之前轮次的对话记录到日志
            if video_id in round1_results:
                round1_data = round1_results[video_id]
                result_log.append(
                    f"Question: What do you see in this video? Can you see the AR effect in the video?\n"
                    f"Answer: {round1_data.get('response', 'N/A')}\n"
                    f"Thinking: {round1_data.get('thinking', 'N/A')}\n"
                    f"{'-' * 20}\n"
                )

            if video_id in round2_results:
                round2_data = round2_results[video_id]
                result_log.append(
                    f"Question: What is the 3D object in the video?\n"
                    f"Answer: {round2_data.get('response', 'N/A')}\n"
                    f"Thinking: {round2_data.get('thinking', 'N/A')}\n"
                    f"{'-' * 20}\n"
                )

            # 评估待处理的AR指标
            for metric in remaining_metrics:
                question = f"Is there any issue with the {metric}? Please clearly state your reason and give your answer in JSON format."

                # 构建消息，包含完整历史
                metric_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The video name is {video_name}. These are frames from a video that I want to analyze."
                            }
                        ]
                    }
                ]

                # 添加图像内容
                for frame in frames:
                    metric_messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame
                        }
                    })

                # 添加round1的对话历史
                if video_id in round1_results:
                    round1_data = round1_results[video_id]
                    metric_messages.append({
                        "role": "user", 
                        "content": "What do you see in this video? Can you see the AR effect in the video?"
                    })

                    metric_messages.append({
                        "role": "assistant", 
                        "content": [
                            {"type": "thinking", "thinking": round1_data.get('thinking', ''), 
                             "signature": round1_data.get('signature', '')},
                            {"type": "text", "text": round1_data.get('response', 'No response available')}
                        ]
                    })

                # 添加round2的对话历史
                if video_id in round2_results:
                    round2_data = round2_results[video_id]
                    metric_messages.append({
                        "role": "user", 
                        "content": "What is the 3D object in the video?"
                    })

                    metric_messages.append({
                        "role": "assistant", 
                        "content": [
                            {"type": "thinking", "thinking": round2_data.get('thinking', ''), 
                             "signature": round2_data.get('signature', '')},
                            {"type": "text", "text": round2_data.get('response', 'No response available')}
                        ]
                    })
                else:
                    # 如果没有round2的历史，但有round1的历史，添加一个假的round2答案
                    if video_id in round1_results:
                        print(f"Warning: No round 2 results found for {video_name}, using placeholder")
                        metric_messages.append({
                            "role": "user", 
                            "content": "What is the 3D object in the video?"
                        })

                        metric_messages.append({
                            "role": "assistant", 
                            "content": "I cannot determine the exact 3D object from the video frames."
                        })

                # 添加当前问题
                metric_messages.append({
                    "role": "user", 
                    "content": question
                })

                # 尝试多次获取有效的JSON响应
                retry_count = 0
                metric_processed = False

                while retry_count < self.MAX_RETRIES and not metric_processed:
                    try:
                        print(f"Processing metric: {metric} (attempt {retry_count+1})")

                        # 发送请求
                        response = self.client.messages.create(
                            model=self.MODEL,
                            max_tokens=self.MAX_TOKENS,
                            messages=metric_messages,
                            system=self.generate_system_prompt(),
                            temperature=1,
                            thinking={
                                "type": "enabled",
                                "budget_tokens": 16000
                            }
                        )

                        # 提取回复和思考过程
                        answer = ""
                        thinking = ""
                        thinking_signature = ""

                        for content in response.content:
                            if content.type == 'text':
                                answer = content.text
                            elif content.type == 'thinking':
                                thinking = content.thinking
                                thinking_signature = content.signature

                        # 更新对话日志
                        current_log_entry = f"Question: {question}\nAnswer: {answer}\nThinking: {thinking}\n{'-' * 20}\n"
                        result_log.append(current_log_entry)

                        # 更新对话日志文件（每个metric完成后立即更新）
                        with open(output_file, "w") as f:
                            f.writelines(result_log)

                        # 提取JSON
                        json_match = re.search(r"\{[^{}]*\}", answer)
                        if json_match:
                            json_str = json_match.group(0)
                            json_data = json.loads(json_str)

                            # 验证必要字段
                            if all(key in json_data for key in ["Video_name", "Metrics", "Issue", "Reason"]):
                                # 确保指标名称是标准化的
                                json_data['Metrics'] = metric
                                # 添加thinking内容
                                json_data['thinking'] = thinking
                                # 添加custom_id
                                json_data['custom_id'] = f"{video_id}_{metric.lower().replace(' ', '_')}"

                                # 立即保存JSON结果（每个metric完成后）
                                with open(json_output_file, "a") as f:
                                    json.dump(json_data, f, indent=4)
                                    f.write("\n")

                                print(f"Successfully processed metric: {metric} for {video_id}")
                                metric_processed = True

                                # 记录处理进度更新
                                processed_videos = self.get_processed_videos()
                                processed_metrics = processed_videos.get(video_id, set())
                                processed_metrics.add(metric)
                                remaining = set(self.METRICS) - processed_metrics

                                print(f"Updated processing status for {video_id}")
                                if remaining:
                                    print(f"Remaining metrics: {', '.join(remaining)}")
                                else:
                                    print(f"All metrics for {video_id} are now processed")

                            else:
                                print(f"Warning: Missing required fields in JSON response for {metric}")
                        else:
                            print(f"Warning: Could not extract JSON from response for {metric}")

                        if not metric_processed:
                            print(f"Retrying metric {metric} (attempt {retry_count+2}/{self.MAX_RETRIES})")
                            retry_count += 1
                            time.sleep(2)  # 短暂等待后重试

                    except Exception as e:
                        print(f"Error processing {metric}: {e}")
                        traceback.print_exc()
                        retry_count += 1
                        time.sleep(5)  # 错误后等待时间更长

                if not metric_processed:
                    print(f"Failed to get valid JSON response for {metric} after {self.MAX_RETRIES} attempts")

                # 每个指标处理完后等待一小段时间，避免API限制
                time.sleep(2)

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            traceback.print_exc()

            # 记录错误
            error_file = self.output_dir / "logs" / "errors.txt"
            with open(error_file, "a") as f:
                f.write(f"Error processing video: {video_path}. Error: {str(e)}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-"*50 + "\n")

    def check_round_files_exist(self) -> bool:
        """检查是否存在前两轮的结果文件"""
        round1_file = self.output_dir / "round1" / "results.json"
        round2_file = self.output_dir / "round2" / "results.json"
        
        if not round1_file.exists():
            print(f"Warning: Round 1 results file not found: {round1_file}")
            print("You may need to run batch processing for round 1 first")
            return False
            
        if not round2_file.exists():
            print(f"Warning: Round 2 results file not found: {round2_file}")
            print("You may need to run batch processing for round 2 first")
            return False
            
        return True
    
    def process_videos(self, video_dir: str, mode: str = "discard"):
        """主处理函数，处理所有视频。"""
        self.initialize_video_mapping(video_dir)
        
        # 检查是否存在前两轮的结果文件
        if not self.check_round_files_exist():
            print("Continuing without round 1/2 results. Some videos may be skipped or have incomplete context.")
            
        # 获取视频文件列表
        video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.mov'))]
        
        # 获取已经处理过的视频
        processed_videos_info = self.get_processed_videos()
        
        # 如果需要继续处理
        if mode == "continue":
            # 过滤掉已经完成所有指标的视频
            filtered_video_files = []
            for vf in video_files:
                video_name = os.path.basename(vf)
                video_id = self.get_video_id(video_name)
                processed_metrics = processed_videos_info.get(video_id, set())
                if len(processed_metrics) < len(self.METRICS):
                    filtered_video_files.append(vf)
                else:
                    print(f"Skipping {video_id} - all metrics already processed")
            
            video_files = filtered_video_files
        
        total_videos = len(video_files)
        print(f"Found {total_videos} videos to process")
        
        # 清理输出文件（如果是discard模式）
        if mode == "discard":
            json_output_file = self.output_dir / "json" / "label_claude_thinking.json"
            if json_output_file.exists():
                print(f"Backing up existing output file to {json_output_file}.bak")
                # 备份现有的输出文件
                if os.path.exists(json_output_file):
                    backup_file = f"{json_output_file}.bak"
                    i = 1
                    while os.path.exists(backup_file):
                        backup_file = f"{json_output_file}.bak.{i}"
                        i += 1
                    os.rename(json_output_file, backup_file)
                
                # 创建新的空文件
                with open(json_output_file, "w") as f:
                    pass
        
        # 逐个处理视频
        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            video_id = self.get_video_id(video_name)
            processed_metrics = processed_videos_info.get(video_id, set()) 
            
            print(f"\nProcessing video {i}/{total_videos}: {video_id}")
            print(f"Already processed metrics: {', '.join(processed_metrics) if processed_metrics else 'None'}")
            
            self.process_video(video_path)
            # 给API一些休息时间
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="Process AR videos with Claude 3.7 and thinking.")
    parser.add_argument(
        "--mode", 
        choices=["discard", "continue"], 
        default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos."
    )
    parser.add_argument(
        "--video_dir", 
        type=str, 
        default="../video_clips_6s_2",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="5_output",
        help="Directory to save output files (should contain round1 and round2 subdirectories with results.json)"
    )
    parser.add_argument(
        "--specific_video",
        type=str,
        default=None,
        help="Only process a specific video by name (optional)"
    )
    
    args = parser.parse_args()
    
    processor = ARVideoProcessor(output_dir=args.output_dir)
    
    if args.specific_video:
        # 只处理指定的视频
        video_path = os.path.join(args.video_dir, args.specific_video)
        if os.path.exists(video_path):
            print(f"Processing single video: {args.specific_video}")
            processor.initialize_video_mapping(args.video_dir)
            processor.process_video(video_path)
        else:
            print(f"Error: Video not found: {video_path}")
    else:
        # 处理所有视频
        processor.process_videos(args.video_dir, args.mode)

if __name__ == "__main__":
    main()