"""
python 5_label_claude_batch.py --video_dir ../video_clips_6s_2_test --output_dir 5_output --start_round 1
"""
import os
import cv2
import base64
import time
import json
import re
import argparse
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

load_dotenv()

class ARVideoProcessor:
    MODEL = "claude-3-5-sonnet-20241022"
    FRAME_INTERVAL = 10  # Frame sampling interval
    POLLING_INTERVAL = 60  # seconds between status checks
    MAX_TOKENS = 4096
    BATCH_SIZE = 3
    MAX_WAIT_TIME = 30*60  # 20 minutes timeout
    MAX_RETRIES = 3  # maximum number of retry attempts
    
    METRICS = [
        "Object Placement",
        "Occlusion",
        "Object Movement",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    ]

    def __init__(self, output_dir: str = "output"):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.video_mapping = {}  # 视频名称到编号的映射
        self.reverse_mapping = {}  # 编号到视频名称的映射
        
    def setup_directories(self):
        """Create necessary output directories if they don't exist."""
        directories = ['round1', 'round2', 'json', 'logs', 'mapping']
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def initialize_video_mapping(self, video_dir: str):
        """为所有视频创建编号映射."""
        video_files = sorted([f for f in os.listdir(video_dir) 
                            if f.endswith(('.mp4', '.mov'))])
        
        # 创建新的映射
        for idx, video_name in enumerate(video_files, 1):
            video_id = f"v{idx:04d}"  # 使用4位数字，前导零填充
            self.video_mapping[video_name] = video_id
            self.reverse_mapping[video_id] = video_name
        
        # 保存映射到文件
        mapping_file = self.output_dir / "mapping" / "video_mapping.json"
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
    
    def read_video_frames(self, video_path: str) -> List[str]:
        """Read and sample video frames, converting to base64."""
        video = cv2.VideoCapture(video_path)
        base64_frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Resize frame if necessary
            resized_frame = self.resize_frame(frame)

            # Convert to JPEG and then to base64
            _, buffer = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        # Sample frames at regular intervals
        sampled_frames = base64_frames[0::self.FRAME_INTERVAL]
        print(f"{len(base64_frames)} frames read from {video_path}, {len(sampled_frames)} frames sampled")

        return sampled_frames
    
    def process_videos_in_batches(self, video_paths: List[str], round_num: int) -> List[Dict]:
        """Processes videos in batches with timeout and retry logic."""
        all_results = []

        # 将视频路径分成小批次
        for i in range(0, len(video_paths), self.BATCH_SIZE):
            batch_paths = video_paths[i:i + self.BATCH_SIZE]
            print(f"\nProcessing batch {i//self.BATCH_SIZE + 1} ({len(batch_paths)} videos)")

            # 生成请求
            if round_num == 1:
                requests = self.create_round1_batch(batch_paths)
            elif round_num == 2:
                requests = self.create_round2_batch(batch_paths)
            else:
                requests = []
                for metric in self.METRICS:
                    metric_requests = self.create_round3_batch(batch_paths, metric)
                    requests.extend(metric_requests)

            if not requests:
                print("No requests generated for this batch")
                continue

            # 批次处理，带重试逻辑
            for retry in range(self.MAX_RETRIES):
                try:
                    print(f"Starting batch processing with {len(requests)} requests (attempt {retry + 1})")
                    batch = self.client.messages.batches.create(requests=requests)

                    # 等待批次完成，加入超时检查
                    start_time = time.time()
                    while True:
                        batch_status = self.client.messages.batches.retrieve(batch.id)
                        if batch_status.processing_status == "ended":
                            results = list(self.client.messages.batches.results(batch.id))
                            json_results, _ = self.process_results(round_num, results)
                            if json_results:
                                all_results.extend(json_results)
                            break

                        # 检查是否超时 (20分钟)
                        if time.time() - start_time > self.MAX_WAIT_TIME:  # 20 * 60 seconds
                            print(f"Batch {batch.id} timed out after 20 minutes")
                            # 尝试取消当前批次
                            try:
                                self.client.messages.batches.cancel(batch.id)
                            except Exception as e:
                                print(f"Error cancelling batch: {str(e)}")
                            raise TimeoutError("Batch processing timed out")

                        print(f"Batch {batch.id} is still processing...")
                        time.sleep(self.POLLING_INTERVAL)

                    # 如果成功处理完成，退出重试循环
                    break

                except (TimeoutError, Exception) as e:
                    print(f"Error during batch processing (attempt {retry + 1}): {str(e)}")
                    if retry == self.MAX_WAIT_TIME - 1:
                        print(f"Failed to process batch after {self.MAX_WAIT_TIME} attempts")
                    else:
                        print("Retrying batch processing...")
                        time.sleep(self.POLLING_INTERVAL)  # 等待一段时间后重试

            # 在处理下一批之前等待，避免API限制
            if i + self.BATCH_SIZE < len(video_paths):
                print("Waiting before processing next batch...")
                time.sleep(self.POLLING_INTERVAL)

        return all_results

    def load_round_results(self, round_num: int) -> Dict:
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
        return {}

    def get_processed_videos(self, round_num: int) -> Set[str]:
        """Get set of video names that have been processed in a specific round."""
        results = self.load_round_results(round_num)
        return set(results.keys())

    # Update the create_roundX_batch methods to use sanitized custom_id consistently
    def create_round1_batch(self, video_paths: List[str]) -> List[Request]:
        """Create batch requests for the first round of conversation."""
        requests = []
        processed_videos = self.get_processed_videos(1)

        for video_path in video_paths:
            video_name = Path(video_path).name
            video_id = self.get_video_id(video_name)

            if video_id in processed_videos:
                print(f"Skipping {video_id} - already processed in round 1")
                continue

            print(f"Reading frames for {video_name}...")
            frames = self.read_video_frames(video_path)

            content = [{"type": "text", "text": f"The video name is {video_name}. These are frames from a video that I want to analyze."}]
            content.extend([{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame
                }
            } for frame in frames])
            content.append({
                "type": "text",
                "text": "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video?"
            })

            requests.append(Request(
                custom_id=video_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    messages=[{"role": "user", "content": content}],
                    system=self.generate_system_prompt()
                )
            ))
        return requests
    
    def create_round2_batch(self, video_paths: List[str]) -> List[Request]:
        """Create batch requests for the second round of conversation."""
        requests = []
        processed_videos = self.get_processed_videos(2)
        round1_results = self.load_round_results(1)

        for video_path in video_paths:
            video_name = Path(video_path).name
            video_id = self.get_video_id(video_name)

            if video_id in processed_videos:
                print(f"Skipping {video_name} (ID: {video_id}) - already processed in round 2")
                continue

            # 读取视频帧，无论是否有round1结果
            print(f"Reading frames for {video_name}...")
            frames = self.read_video_frames(video_path)

            # 准备基础内容，包含视频帧
            base_content = [
                {"type": "text", "text": f"The video name is {video_name}. These are frames from a video that I want to analyze."}
            ]
            base_content.extend([{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame
                }
            } for frame in frames])

            # 构建消息历史
            messages = []
            if video_id in round1_results:
                # 如果有第一轮结果，添加完整对话历史
                messages = [
                    {"role": "user", "content": base_content + [{"type": "text", "text": "What do you see in this video?"}]},
                    {"role": "assistant", "content": round1_results[video_id]},
                    {"role": "user", "content": "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it."}
                ]
            else:
                # 如果没有第一轮结果，只包含视频帧和当前问题
                messages = [
                    {"role": "user", "content": base_content + [{"type": "text", "text": "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it."}]}
                ]

            requests.append(Request(
                custom_id=video_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    messages=messages,
                    system=self.generate_system_prompt()
                )
            ))

        return requests
    
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

    def create_round3_batch(self, video_paths: List[str], metric: str) -> List[Request]:
        """Create batch requests for the third round of conversation."""
        requests = []
        round1_results = self.load_round_results(1)
        round2_results = self.load_round_results(2)

        # 获取已处理的视频和metrics
        processed_items = self.get_processed_round3_items()

        # 为metrics创建简短的标识符
        metric_abbrev = {
            "Object Placement": "op",
            "Object Movement": "om",
            "Occlusion": "oc",
            "Lighting": "lt",
            "Visual Artifacts and Rendering Issues": "ar",
            "Black Screen": "bs"
        }

        for video_path in video_paths:
            video_name = Path(video_path).name
            video_id = self.get_video_id(video_name)

            # 检查是否已处理此视频的此metric
            if video_id in processed_items and metric in processed_items[video_id]:
                print(f"Skipping {video_id} - {metric} (already processed)")
                continue

            # 读取视频帧
            print(f"Reading frames for {video_name}...")
            frames = self.read_video_frames(video_path)

            # 准备基础内容，包含视频帧
            base_content = [
                {"type": "text", "text": f"The video name is {video_name}. These are frames from a video that I want to analyze."}
            ]
            base_content.extend([{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame
                }
            } for frame in frames])

            # 构建消息历史
            messages = []
            if video_id in round1_results and video_id in round2_results:
                # 完整的对话历史
                messages = [
                    {"role": "user", "content": base_content + [{"type": "text", "text": "What do you see in this video?"}]},
                    {"role": "assistant", "content": round1_results[video_id]},
                    {"role": "user", "content": "What is the 3D object in the video?"},
                    {"role": "assistant", "content": round2_results[video_id]},
                    {"role": "user", "content": f"Is there any issue with the {metric}? Please clearly state your reason and give your answer in JSON format."}
                ]
            else:
                # 只包含视频帧和当前问题
                messages = [
                    {"role": "user", "content": base_content + [{"type": "text", "text": f"Is there any issue with the {metric}? Please clearly state your reason and give your answer in JSON format."}]}
                ]

            metric_id = f"{video_id}_{metric_abbrev[metric]}"
            requests.append(Request(
                custom_id=metric_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    messages=messages,
                    system=self.generate_system_prompt()
                )
            ))

        return requests

    """def wait_for_batch_completion(self, batch_id: str) -> List[Any]:
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return list(self.client.messages.batches.results(batch_id))
            print(f"Batch {batch_id} is still processing...")
            time.sleep(self.POLLING_INTERVAL)"""
            
    # Add this new method for error handling
    def save_error(self, round_num: int, custom_id: str, error_info: Dict):
        """Save error information to the corresponding round's error file."""
        error_file = self.output_dir / f"round{round_num}" / "errors.json"
        errors = {}
        
        # Load existing errors if file exists
        if error_file.exists():
            with open(error_file, 'r') as f:
                errors = json.load(f)
        
        # Add new error
        errors[custom_id] = error_info
        
        # Save updated errors
        with open(error_file, 'a') as f:
            json.dump(errors, f, indent=4)

    def save_round_results(self, round_num: int, results: Dict):
        """Save results for a specific round with proper JSON handling."""
        results_file = self.output_dir / f"round{round_num}" / "results.json"
        
        # Load existing results if file exists
        existing_results = {}
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing results in {results_file}")
                # Backup corrupted file
                backup_file = results_file.with_suffix('.json.bak')
                results_file.rename(backup_file)
        
        # Merge new results with existing results
        existing_results.update(results)
        
        # Write all results back to file
        with open(results_file, 'w') as f:
            json.dump(existing_results, f, indent=4)

    def save_complete_conversation(self, video_id: str, conversation: List[Dict]):
        """Save complete conversation for a video."""
        original_video_name = self.get_video_name(video_id)
        with open(self.output_dir / "logs" / "complete_conversations.txt", 'a') as f:
            f.write(f"\nProcessing video: {original_video_name} (ID: {video_id})\n\n")
            for msg in conversation:
                if msg["role"] == "user":
                    f.write(f"Question: {msg['content']}\n")
                else:
                    f.write(f"Answer: {msg['content']}\n")
                f.write("-" * 20 + "\n")

    def save_json_results(self, results: List[Dict]):
        """Save all JSON results to a single file."""
        # 确保 json 目录存在
        output_file = self.output_dir / "json" / "label_claude.json"

        processed_results = []
        for result in results:
            # 从 custom_id 中提取视频ID和指标类型
            # custom_id 格式应该是: 'v0001_op' 或类似格式
            try:
                # 分割 custom_id 得到视频ID (v0001)
                video_id = result.get('custom_id', '').split('_')[0]

                # 通过映射获取原始视频名称
                video_name = self.get_video_name(video_id)

                # 创建新的结果字典，添加准确的视频名称
                processed_result = result.copy()
                processed_result['Video_name'] = video_name
                processed_results.append(processed_result)

            except Exception as e:
                print(f"Error processing result with custom_id {result.get('custom_id', 'unknown')}: {str(e)}")
                # 如果出错，仍然保存原始结果
                processed_results.append(result)

        # 保存处理后的结果
        with open(output_file, 'a') as f:
            for result in processed_results:
                json.dump(result, f, indent=4)
                f.write('\n')

    @staticmethod
    def generate_system_prompt() -> str:
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

    # Modify the process_results method to handle errors
    def process_results(self, round_num: int, results: List[Any], metric: str = None) -> Tuple[List[Dict], Dict]:
        """Process batch results based on the round number."""
        successful_results = {}
        json_results = []

        for result in results:
            if result.result.type == 'succeeded':
                response_text = result.result.message.content[0].text

                if round_num in [1, 2]:
                    successful_results[result.custom_id] = response_text
                else:  # round 3
                    try:
                        json_match = re.search(r"\{[^{}]*\}", response_text)
                        if json_match:
                            json_data = json.loads(json_match.group(0))
                            # 添加 custom_id 到 JSON 数据中
                            json_data['custom_id'] = result.custom_id
                            json_results.append(json_data)
                    except (json.JSONDecodeError, AttributeError) as e:
                        self.save_error(round_num, result.custom_id, {
                            "error_type": "json_parse_error",
                            "error_message": str(e),
                            "response_text": response_text
                        })
            else:
                self.save_error(round_num, result.custom_id, {
                    "error_type": result.result.type,
                    "error_message": result.result.error.message
                })

        if round_num in [1, 2]:
            # Save results to the corresponding round's file
            self.save_round_results(round_num, successful_results)
            return [], successful_results
        else:
            return json_results, {}
    

    def process_videos(self, video_dir: str, start_round: int = 1):
        """Main processing function that handles all rounds of conversation."""
        self.initialize_video_mapping(video_dir)
        
        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.mov'))]
        
        all_json_results = []
        current_round = start_round
        
        # Process Round 1 if needed
        if current_round == 1:
            print("Processing Round 1...")
            self.process_videos_in_batches(video_paths, 1)
            current_round += 1
        
        # Process Round 2 if needed
        if current_round <= 2:
            print("Processing Round 2...")
            self.process_videos_in_batches(video_paths, 2)
            current_round += 1
        
        # Process Round 3 with all metrics in one batch
        if current_round <= 3:
            print("Processing Round 3...")
            json_results = self.process_videos_in_batches(video_paths, 3)
            all_json_results.extend(json_results)
        
        # Save complete conversations for all videos
        for video_path in video_paths:
            video_name = Path(video_path).name
            video_id = self.get_video_id(video_name)
            conversation = []
            
            # Add conversations from all rounds
            self.build_conversation_history(video_id, conversation, start_round)
            
            # Save complete conversation
            if conversation:
                self.save_complete_conversation(video_id, conversation)
        
        # Save all JSON results
        if all_json_results:
            self.save_json_results(all_json_results)
            
    def build_conversation_history(self, video_id: str, conversation: List[Dict], start_round: int):
        """构建完整的对话历史"""
        # Add Round 1 conversation if available
        if start_round == 1:
            round1_results = self.load_round_results(1)
            if video_id in round1_results:
                conversation.extend([
                    {"role": "user", "content": "What do you see in this video?"},
                    {"role": "assistant", "content": round1_results[video_id]}
                ])
        
        # Add Round 2 conversation if available
        if start_round <= 2:
            round2_results = self.load_round_results(2)
            if video_id in round2_results:
                conversation.extend([
                    {"role": "user", "content": "What is the 3D object in the video?"},
                    {"role": "assistant", "content": round2_results[video_id]}
                ])
        
        # Add Round 3 conversations if available
        if start_round <= 3:
            round3_results = self.load_round_results(3) if hasattr(self, 'load_round_results') else {}
            for metric in self.METRICS:
                metric_id = f"{video_id}_{metric.lower().replace(' ', '_')}"
                if metric_id in round3_results:
                    conversation.extend([
                        {"role": "user", "content": f"Is there any issue with the {metric}?"},
                        {"role": "assistant", "content": round3_results[metric_id]}
                    ])

def main():
    parser = argparse.ArgumentParser(description="Process AR videos with Claude batch processing.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--start_round", type=int, default=1, choices=[1, 2, 3],
                      help="Round to start processing from (1, 2, or 3)")
    
    args = parser.parse_args()
    
    processor = ARVideoProcessor(output_dir=args.output_dir)
    processor.process_videos(args.video_dir, args.start_round)

if __name__ == "__main__":
    main()