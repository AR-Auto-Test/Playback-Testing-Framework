"""
python 5_label_claude_batch_thinking.py --video_dir ../video_clips_6s_2 --output_dir 5_output --start_round 1 --batch_num 3
python 5_label_claude_batch_thinking.py --video_dir ../video_clips_6s_3 --output_dir 5_output --start_round 3 --batch_num 3
"""
import os
import cv2
import base64
import time
import json
import re
import argparse
import pickle
import concurrent.futures
import threading
import traceback
import requests
import anthropic
from threading import Lock
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

import ar_prompts_labels

load_dotenv()

class ARVideoProcessor:
    MODEL = "claude-3-7-sonnet-20250219"
    FRAME_INTERVAL = 10  # Frame sampling interval
    POLLING_INTERVAL = 60  # seconds between status checks
    MAX_TOKENS = 20000
    BATCH_SIZE = 3
    MAX_WAIT_TIME = 30*60  # 20 minutes timeout
    MAX_RETRIES = 3  # maximum number of retry attempts
    RETRY_DELAY = 30
    
    # Add these class variables
    TEMP_DIR = f"temp_frames_{FRAME_INTERVAL}"
    
    METRICS = [
        "Object Placement",
        "Occlusion",
        "Object Movement",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    ]

    def __init__(self, output_dir: str = "output", batch_num: int = 5):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=120.0)
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.video_mapping = {}  # 视频名称到编号的映射
        self.reverse_mapping = {}  # 编号到视频名称的映射
        # Create temp directory for frames
        self.temp_frames_dir = Path(self.TEMP_DIR)
        self.temp_frames_dir.mkdir(exist_ok=True)
        self.batch_num = batch_num
        self.batch_lock = Lock()  # 用于控制批次处理的锁
        
    def setup_directories(self):
        """Create necessary output directories if they don't exist."""
        directories = ['round1', 'round2', 'round3', 'json', 'logs', 'mapping']  # 添加 'round3'
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def get_frames_path(self, video_name: str) -> str:
        """Get the path to the cached frames file for a video."""
        video_id = self.get_video_id(video_name)
        return str(self.temp_frames_dir / f"{video_id}_frames.pkl")
    
    def save_frames(self, video_name: str, frames: List[str]) -> None:
        """Save video frames to a temporary file."""
        frames_path = self.get_frames_path(video_name)
        with open(frames_path, 'wb') as f:
            pickle.dump(frames, f)
            
    def load_frames(self, video_name: str) -> List[str]:
        """Load video frames from temporary file if available, otherwise sample them."""
        frames_path = self.get_frames_path(video_name)
        if os.path.exists(frames_path):
            with open(frames_path, 'rb') as f:
                return pickle.load(f)
        return None
            
    def initialize_video_mapping(self, video_dir: str):
        """为所有视频创建编号映射，如果已存在则加载."""
        mapping_file = self.output_dir / "mapping" / "video_mapping.json"

        # 检查是否已存在mapping文件
        if mapping_file.exists():
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                    self.video_mapping = mapping_data.get("video_to_id", {})
                    self.reverse_mapping = mapping_data.get("id_to_video", {})

                print(f"Loaded existing mapping for {len(self.video_mapping)} videos")
                return self.video_mapping
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading mapping file: {str(e)}. Creating new mapping.")

        # 如果没有现有mapping或加载失败，创建新的mapping
        video_files = sorted([f for f in os.listdir(video_dir) 
                            if f.endswith(('.mp4', '.mov'))])

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

        print(f"Created new mapping for {len(video_files)} videos")
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
            #resized_frame = frame

            # Convert to JPEG and then to base64
            _, buffer = cv2.imencode(".jpg", resized_frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()

        # Sample frames at regular intervals
        sampled_frames = base64_frames[0::self.FRAME_INTERVAL]
        print(f"{len(base64_frames)} frames read from {video_path}, {len(sampled_frames)} frames sampled")

        return sampled_frames
    
    def process_videos_in_batches(self, video_paths: List[str], round_num: int) -> List[Dict]:
        all_results = []

        # First determine which videos need processing for this round
        videos_to_process = []
        if round_num == 1:
            processed_videos = self.get_processed_videos(1)
            videos_to_process = [video_path for video_path in video_paths 
                                 if self.get_video_id(Path(video_path).name) not in processed_videos]
        elif round_num == 2:
            # For round 2, we need videos that have completed round 1 but not round 2
            round1_results = self.load_round_results(1)
            processed_videos_round2 = self.get_processed_videos(2)

            videos_to_process = [video_path for video_path in video_paths 
                                 if (self.get_video_id(Path(video_path).name) in round1_results and 
                                     self.get_video_id(Path(video_path).name) not in processed_videos_round2)]
        else:  # round 3
            # For round 3, we need videos that have completed both round 1 and round 2
            round1_results = self.load_round_results(1)
            round2_results = self.load_round_results(2)
            processed_items = self.get_processed_round3_items()

            videos_to_process = []
            for video_path in video_paths:
                video_name = Path(video_path).name
                video_id = self.get_video_id(video_name)

                # Only include if both round 1 and 2 are complete AND any metric is missing
                if (video_id in round1_results and video_id in round2_results and 
                    (video_id not in processed_items or len(processed_items[video_id]) < len(self.METRICS))):
                    videos_to_process.append(video_path)

        print(f"Round {round_num}: {len(videos_to_process)}/{len(video_paths)} videos need processing")

        if not videos_to_process:
            print(f"All eligible videos already processed for round {round_num}. Skipping.")
            return all_results

        # Pre-process: Sample frames for all videos that need processing
        for video_path in videos_to_process:
            video_name = Path(video_path).name
            if not os.path.exists(self.get_frames_path(video_name)):
                print(f"Sampling frames for {video_name}...")
                frames = self.read_video_frames(video_path)
                self.save_frames(video_name, frames)
                print(f"Frames saved for {video_name}")
            else:
                print(f"Frames already exist for {video_name}")

        def process_batch(batch_paths, batch_index, total_batches, progress_lock):
            try:
                # Create batch requests
                requests = self.create_batch_requests(batch_paths, round_num)
                retry_delay = self.RETRY_DELAY

                # Skip if no requests were generated
                if not requests:
                    with progress_lock:
                        print(f"Round {round_num} Batch {batch_index}/{total_batches} - No requests generated (Skipped)")
                    return {
                        "results": [],
                        "status": "skipped"
                    }

                with progress_lock:
                    print(f"Round {round_num} Batch {batch_index}/{total_batches} - Generating {len(requests)} requests")

                for retry in range(self.MAX_RETRIES):    
                    try:
                        # Submit batch
                        batch = self.client.messages.batches.create(requests=requests)

                        with progress_lock:
                            print(f"Round {round_num} Batch {batch_index}/{total_batches} - Successfully submitted (Batch ID {batch.id})")

                        # Break out of retry loop if submission was successful
                        break

                    except Exception as submit_error:
                        if retry < self.MAX_RETRIES - 1:
                            with progress_lock:
                                print(f"Timeout error. Retrying in {retry_delay} seconds... (Attempt {retry+1}/{self.MAX_RETRIES})")
                            time.sleep(retry_delay)
                            # Exponential backoff: double delay with each retry
                            retry_delay *= 2
                        else:
                            # Detailed error handling
                            error_details = {
                                "error_type": type(submit_error).__name__,
                                "error_message": str(submit_error),
                                "traceback": traceback.format_exc()
                            }

                            with progress_lock:
                                print(f"Round {round_num} Batch {batch_index}/{total_batches} - Submission failed")
                                print("Detailed Error Information:")
                                print(json.dumps(error_details, indent=2))

                            return {
                                "results": [],
                                "status": "Submission error",
                                "error_details": error_details
                            }

                # Wait for batch to complete
                start_time = time.time()
                while True:
                    try:
                        batch_status = self.client.messages.batches.retrieve(batch.id)

                        if batch_status.processing_status == "ended":
                            with progress_lock:
                                print(f"Round {round_num} Batch {batch_index}/{total_batches} - Processing completed")

                            try:
                                results = list(self.client.messages.batches.results(batch.id))
                                json_results, _ = self.process_results(round_num, results)

                                with progress_lock:
                                    print(f"Round {round_num} Batch {batch_index}/{total_batches} - Processed {len(json_results)} results")

                                return {
                                    "results": json_results,
                                    "status": "completed"
                                }
                            except Exception as results_error:
                                error_details = {
                                    "error_type": type(results_error).__name__,
                                    "error_message": str(results_error),
                                    "traceback": traceback.format_exc()
                                }
                                with progress_lock:
                                    print("Error processing batch results:")
                                    print(json.dumps(error_details, indent=2))

                                return {
                                    "results": [],
                                    "status": "results_error",
                                    "error_details": error_details
                                }

                        if time.time() - start_time > self.MAX_WAIT_TIME:
                            with progress_lock:
                                print(f"Round {round_num} Batch {batch_index}/{total_batches} - Timed out after {self.MAX_WAIT_TIME / 60} minutes")
                            self.client.messages.batches.cancel(batch.id)
                            return {
                                "results": [],
                                "status": "timeout"
                            }

                        with progress_lock:
                            print(f"Round {round_num} Batch {batch_index}/{total_batches} - Still processing... (Status {batch_status.processing_status})")
                        time.sleep(self.POLLING_INTERVAL)

                    except Exception as processing_error:
                        error_details = {
                            "error_type": type(processing_error).__name__,
                            "error_message": str(processing_error),
                            "traceback": traceback.format_exc()
                        }
                        with progress_lock:
                            print(f"Round {round_num} Batch {batch_index}/{total_batches} - Processing error")
                            print("Detailed Error Information:")
                            print(json.dumps(error_details, indent=2))

                        return {
                            "results": [],
                            "status": "processing_error",
                            "error_details": error_details
                        }

            except Exception as e:
                error_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                with progress_lock:
                    print(f"Round {round_num} Batch {batch_index}/{total_batches} - Unexpected error")
                    print("Detailed Error Information:")
                    print(json.dumps(error_details, indent=2))

                return {
                    "results": [],
                    "status": "unexpected_error",
                    "error_details": error_details
                }

        # Divide videos into batches
        video_batches = [videos_to_process[i:i + self.BATCH_SIZE] for i in range(0, len(videos_to_process), self.BATCH_SIZE)]
        total_batches = len(video_batches)
        completed_batches = 0

        # Progress lock for thread-safe printing
        progress_lock = threading.Lock()

        with progress_lock:
            print(f"Round {round_num} Total batches to process - {total_batches}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            # Process batches using ThreadPoolExecutor
            futures = {
                executor.submit(
                    process_batch, 
                    batch, 
                    batch_idx + 1, 
                    total_batches, 
                    progress_lock
                ): batch 
                for batch_idx, batch in enumerate(video_batches)
            }

            for future in concurrent.futures.as_completed(futures):
                batch_result = future.result()

                # Add results regardless of batch status
                if batch_result['results']:
                    all_results.extend(batch_result['results'])

                with progress_lock:
                    completed_batches += 1
                    percentage = (completed_batches / total_batches) * 100
                    print(f"Round {round_num} Completed {completed_batches}/{total_batches} batches ({percentage:.1f}%)")

                    # Print extra info for non-successful batches
                    if batch_result['status'] != 'completed':
                        print(f"Batch status: {batch_result['status']}")
                        if 'error_details' in batch_result:
                            print("Error details:")
                            print(json.dumps(batch_result['error_details'], indent=2))

                # Optional: Add slight delay between batches
                time.sleep(self.POLLING_INTERVAL / 300)

            with progress_lock:
                print(f"Round {round_num} All batches processed. Total results {len(all_results)}")
            return all_results
    
    def create_batch_requests(self, batch_paths: List[str], round_num: int) -> List[Request]:
        """根据轮次创建批次请求"""
        if round_num == 1:
            return self.create_round1_batch(batch_paths)
        elif round_num == 2:
            return self.create_round2_batch(batch_paths)
        else:
            requests = []
            for metric in self.METRICS:
                metric_requests = self.create_round3_batch(batch_paths, metric)
                requests.extend(metric_requests)
            return requests

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

        # Get conversation prompts from ar_prompts_labels
        conversation_prompts = ar_prompts_labels.generate_conversation_prompts()

        for video_path in video_paths:
            video_name = Path(video_path).name
            video_id = self.get_video_id(video_name)

            # Skip if already processed
            if video_id in processed_videos:
                print(f"Skipping {video_id} - already processed in round 1")
                continue

            # Load frames from cache
            frames = self.load_frames(video_name)
            if not frames:
                print(f"Warning: No cached frames found for {video_name}")
                print(f"Reading frames for {video_name}...")
                frames = self.read_video_frames(video_path)
                self.save_frames(video_name, frames)

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
                "text": conversation_prompts[0]
            })

            requests.append(Request(
                custom_id=video_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    temperature=1,
                    messages=[{"role": "user", "content": content}],
                    system=self.generate_system_prompt(),
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 16000
                    }
                )
            ))
        return requests
    
    def create_round2_batch(self, video_paths: List[str]) -> List[Request]:
        """Create batch requests for the second round of conversation."""
        requests = []
        processed_videos = self.get_processed_videos(2)
        round1_results = self.load_round_results(1)

        # Get conversation prompts from ar_prompts_labels
        conversation_prompts = ar_prompts_labels.generate_conversation_prompts()

        for video_path in video_paths:
            video_name = Path(video_path).name
            video_id = self.get_video_id(video_name)

            # Skip if already processed
            if video_id in processed_videos:
                print(f"Skipping {video_name} (ID: {video_id}) - already processed in round 2")
                continue

            # Load frames from cache
            frames = self.load_frames(video_name)
            if not frames:
                print(f"Warning: No cached frames found for {video_name}")
                print(f"Reading frames for {video_name}...")
                frames = self.read_video_frames(video_path)
                self.save_frames(video_name, frames)

            # Prepare base content with video frames
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

            # Build message history
            messages = []
            if video_id in round1_results:
                # If there are round 1 results, add complete conversation history
                messages = [
                    {"role": "user", "content": base_content + [{"type": "text", "text": conversation_prompts[0]}]},
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "thinking", "thinking": round1_results[video_id].get('thinking', ''), "signature": round1_results[video_id].get('signature', '')},
                            {"type": "text", "text": round1_results[video_id]['response']}
                        ]
                    },
                    {"role": "user", "content": [{"type": "text", "text": conversation_prompts[1]}]}
                ]
            else:
                # If no round 1 results, only include video frames and current question
                messages = [
                    {"role": "user", "content": base_content + [{"type": "text", "text": conversation_prompts[1]}]}
                ]

            requests.append(Request(
                custom_id=video_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    temperature=1,
                    messages=messages,
                    system=self.generate_system_prompt(),
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 16000
                    }
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
        """Create batch requests for the third round of conversation with token counting."""
        requests = []
        round1_results = self.load_round_results(1)
        round2_results = self.load_round_results(2)

        # Get already processed videos and metrics
        processed_items = self.get_processed_round3_items()

        # Get metrics and conversation prompts from ar_prompts_labels
        metrics = ar_prompts_labels.get_metrics()
        conversation_prompts = ar_prompts_labels.generate_conversation_prompts()

        # Create short identifiers for metrics
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

            # Check if this video's metric has already been processed
            if video_id in processed_items and metric in processed_items[video_id]:
                print(f"Skipping {video_id} - {metric} (already processed)")
                continue

            # Load frames from cache
            frames = self.load_frames(video_name)
            if not frames:
                print(f"Warning: No cached frames found for {video_name}")
                print(f"Reading frames for {video_name}...")
                frames = self.read_video_frames(video_path)
                self.save_frames(video_name, frames)

            # Prepare base content with video frames
            original_frame_count = len(frames)
            current_frames = frames.copy()  # Use copy for potential frame adjustment

            # Build message history - always use complete history
            system_prompt = self.generate_system_prompt()
            thinking_budget = 16000  # Default thinking budget

            # Function to rebuild messages with frames
            def rebuild_messages_with_frames(current_frames):
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
                } for frame in current_frames])

                # Get metric index to get the right conversation prompt
                metric_index = metrics.index(metric)
                prompt_index = metric_index + 2  # first 2 are the initial questions

                if video_id in round1_results and video_id in round2_results:
                    # Complete conversation history
                    return [
                        {"role": "user", "content": base_content + [{"type": "text", "text": conversation_prompts[0]}]},
                        {
                            "role": "assistant", 
                            "content": [
                                {"type": "thinking", "thinking": round1_results[video_id].get('thinking', ''), "signature": round1_results[video_id].get('signature', '')},
                                {"type": "text", "text": round1_results[video_id]['response']}
                            ]
                        },
                        {"role": "user", "content": [{"type": "text", "text": conversation_prompts[1]}]},
                        {
                            "role": "assistant", 
                            "content": [
                                {"type": "thinking", "thinking": round2_results[video_id].get('thinking', ''), "signature": round2_results[video_id].get('signature', '')},
                                {"type": "text", "text": round2_results[video_id]['response']}
                            ]
                        },
                        {"role": "user", "content": [{"type": "text", "text": conversation_prompts[prompt_index]}]}
                    ]
                else:
                    # If no history results, only include video frames and current question
                    return [
                        {"role": "user", "content": base_content + [{"type": "text", "text": conversation_prompts[prompt_index]}]}
                    ]

            # Initial message building
            messages = rebuild_messages_with_frames(current_frames)

            metric_id = f"{video_id}_{metric_abbrev[metric]}"
            requests.append(Request(
                custom_id=metric_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    temperature=1,
                    messages=messages,
                    system=system_prompt,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": thinking_budget
                    }
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
        #errors = {}
        
        # Load existing errors if file exists
        #if error_file.exists():
        #    with open(error_file, 'r') as f:
        #        errors = json.load(f)
        
        # Add new error
        #errors[custom_id] = error_info
        
        # Save updated errors
        with open(error_file, 'a') as f:
            json.dump(error_info, f, indent=4)

            
    def save_round_results(self, round_num: int, results: Dict):
        """Save results for a specific round with proper JSON handling."""
        results_file = self.output_dir / f"round{round_num}" / "results.json"

        # 加载现有结果
        existing_results = {}
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing results in {results_file}")
                # 备份损坏的文件
                backup_file = results_file.with_suffix('.json.bak')
                results_file.rename(backup_file)

        # 合并新结果
        existing_results.update(results)

        # 写入所有结果
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
        return ar_prompts_labels.generate_system_prompt()

    # Modify the process_results method to handle errors
    def process_results(self, round_num: int, results: List[Any], metric: str = None) -> Tuple[List[Dict], Dict]:
        """Process batch results based on the round number."""
        successful_results = {}
        json_results = []

        for result in results:
            if result.result.type == 'succeeded':
                # 获取消息内容列表
                message_contents = result.result.message.content

                # 分别处理不同类型的内容块
                response_text = ""
                thinking_text = ""
                thinking_signature = ""

                for content_block in message_contents:
                    if content_block.type == 'text':
                        response_text = content_block.text
                    elif content_block.type == 'thinking':
                        thinking_text = content_block.thinking
                        thinking_signature = content_block.signature  # 新增：获取签名

                # 无论是哪一轮，都保存响应文本和thinking (这是关键修改)
                successful_results[result.custom_id] = {
                    'response': response_text,
                    'thinking': thinking_text,
                    'signature': thinking_signature
                }

                if round_num >= 3:  # round 3
                    try:
                        # 尝试解析 JSON 部分
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
                            "response_text": response_text,
                            "thinking_text": thinking_text,
                            "thinking_signature": thinking_signature
                        })
            else:
                #print(result)
                self.save_error(round_num, result.custom_id, {
                    "error_type": result.result.type,
                    "error_message": result.result.error.error.message
                })

        # 保存所有轮次的原始响应
        self.save_round_results(round_num, successful_results)

        # 第3轮还需要返回解析的JSON结果
        if round_num >= 3:
            return json_results, successful_results
        else:
            return [], successful_results
    

    def process_videos(self, video_dir: str, start_round: int = 1):
        """Main processing function that handles all rounds of conversation."""
        self.initialize_video_mapping(video_dir)

        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.mov'))]

        all_json_results = []
        current_round = start_round

        # Process Round 1 if needed
        if current_round == 1:
            print("\nProcessing Round 1...")
            # Only process videos that have not been completed
            round1_processed = self.get_processed_videos(1)
            videos_to_process = [v for v in video_paths if self.get_video_id(Path(v).name) not in round1_processed]
            if videos_to_process:
                print(f"Found {len(videos_to_process)} videos to process in Round 1")
                self.process_videos_in_batches(video_paths, 1)
            else:
                print("All videos have already been processed in Round 1")
            current_round += 1

        # Process Round 2 if needed
        if current_round <= 2:
            print("\nProcessing Round 2...")
            # Only process videos that have completed round 1 but not round 2
            round1_results = self.load_round_results(1)
            round2_processed = self.get_processed_videos(2)

            # Calculate how many videos are eligible for round 2
            eligible_for_round2 = [v for v in video_paths 
                                  if (self.get_video_id(Path(v).name) in round1_results and 
                                      self.get_video_id(Path(v).name) not in round2_processed)]

            if eligible_for_round2:
                print(f"Found {len(eligible_for_round2)} videos eligible for Round 2")
                self.process_videos_in_batches(video_paths, 2)
            else:
                print("No videos eligible for Round 2 processing")
            current_round += 1

        # Process Round 3 with all metrics
        if current_round <= 3:
            print("\nProcessing Round 3...")
            # Find videos that have completed both round 1 and round 2
            round1_results = self.load_round_results(1)
            round2_results = self.load_round_results(2)
            processed_items = self.get_processed_round3_items()

            # Calculate metrics to process
            metrics_to_process = 0
            eligible_videos = []

            for video_path in video_paths:
                video_name = Path(video_path).name
                video_id = self.get_video_id(video_name)

                # Only include videos that have completed both round 1 and 2
                if video_id in round1_results and video_id in round2_results:
                    eligible_videos.append(video_path)

                    # Count missing metrics
                    if video_id not in processed_items:
                        metrics_to_process += len(self.METRICS)
                    else:
                        metrics_to_process += len(self.METRICS) - len(processed_items[video_id])

            if metrics_to_process > 0:
                print(f"Found {len(eligible_videos)} videos eligible for Round 3 with {metrics_to_process} metric evaluations to complete")
                json_results = self.process_videos_in_batches(video_paths, 3)
                all_json_results.extend(json_results)
            else:
                if eligible_videos:
                    print("All metric evaluations have already been completed for eligible videos in Round 3")
                else:
                    print("No videos eligible for Round 3 processing (missing Round 1 or 2 results)")

        # Calculate and display stats on processing status
        self.display_processing_stats(video_paths)

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
            
    def display_processing_stats(self, video_paths: List[str]):
        """Display statistics about the processing status of all videos."""
        total_videos = len(video_paths)
        round1_results = self.load_round_results(1)
        round2_results = self.load_round_results(2)
        processed_items = self.get_processed_round3_items()

        # Count completed videos for each round
        round1_completed = len(round1_results)
        round2_completed = len(round2_results)

        # Count videos with all metrics completed
        round3_fully_completed = 0
        round3_partially_completed = 0

        for video_path in video_paths:
            video_id = self.get_video_id(Path(video_path).name)
            if video_id in processed_items:
                if len(processed_items[video_id]) == len(self.METRICS):
                    round3_fully_completed += 1
                elif len(processed_items[video_id]) > 0:
                    round3_partially_completed += 1

        # Print statistics
        print("\n" + "="*50)
        print("Processing Statistics:")
        print("-"*50)
        print(f"Total videos: {total_videos}")
        print(f"Round 1 completed: {round1_completed}/{total_videos} ({round1_completed/total_videos*100:.2f}%)")
        print(f"Round 2 completed: {round2_completed}/{total_videos} ({round2_completed/total_videos*100:.2f}%)")
        print(f"Round 3 fully completed: {round3_fully_completed}/{total_videos} ({round3_fully_completed/total_videos*100:.2f}%)")
        print(f"Round 3 partially completed: {round3_partially_completed}/{total_videos}")

        # Calculate remaining work
        round1_remaining = total_videos - round1_completed
        round2_remaining = total_videos - round2_completed
        round3_remaining = total_videos - round3_fully_completed

        print("\nRemaining work:")
        print(f"Videos waiting for Round 1: {round1_remaining}")
        print(f"Videos waiting for Round 2: {round2_remaining}")
        print(f"Videos waiting for Round 3 completion: {round3_remaining}")

        # Calculate metrics statistics
        if processed_items:
            total_metrics = len(self.METRICS) * total_videos
            completed_metrics = sum(len(metrics) for metrics in processed_items.values())
            print(f"\nMetrics completed: {completed_metrics}/{total_metrics} ({completed_metrics/total_metrics*100:.2f}%)")

            # Per-metric statistics
            metric_counts = {metric: 0 for metric in self.METRICS}
            for video_id, metrics in processed_items.items():
                for metric in metrics:
                    if metric in metric_counts:
                        metric_counts[metric] += 1

            print("\nPer-metric completion:")
            for metric, count in metric_counts.items():
                print(f"  {metric}: {count}/{total_videos} ({count/total_videos*100:.2f}%)")

        print("="*50)
            
    def build_conversation_history(self, video_id: str, conversation: List[Dict], start_round: int):
        """构建完整的对话历史"""
        # Add Round 1 conversation if available
        if start_round <= 1:
            round1_results = self.load_round_results(1)
            if video_id in round1_results:
                conversation.extend([
                    {"role": "user", "content": "What do you see in this video?"},
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "thinking", "thinking": round1_results[video_id].get('thinking', ''), "signature": round1_results[video_id].get('signature', '')},
                            {"type": "text", "text": round1_results[video_id]['response']}
                        ]
                    }
                ])

        # Add Round 2 conversation if available
        if start_round <= 2:
            round2_results = self.load_round_results(2)
            if video_id in round2_results:
                conversation.extend([
                    {"role": "user", "content": "What is the 3D object in the video?"},
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "thinking", "thinking": round2_results[video_id].get('thinking', ''), "signature": round2_results[video_id].get('signature', '')},
                            {"type": "text", "text": round2_results[video_id]['response']}
                        ]
                    }
                ])

        # Add Round 3 conversations if available
        if start_round <= 3:
            round3_results = self.load_round_results(3)
            for metric in self.METRICS:
                metric_abbrev = {
                    "Object Placement": "op",
                    "Object Movement": "om",
                    "Occlusion": "oc",
                    "Lighting": "lt",
                    "Visual Artifacts and Rendering Issues": "ar",
                    "Black Screen": "bs"
                }
                metric_id = f"{video_id}_{metric_abbrev[metric]}"
                if metric_id in round3_results:
                    conversation.extend([
                        {"role": "user", "content": f"Is there any issue with the {metric}?"},
                        {
                            "role": "assistant", 
                            "content": [
                                {"type": "thinking", "thinking": round3_results[metric_id].get('thinking', ''), "signature": round3_results[metric_id].get('signature', '')},
                                {"type": "text", "text": round3_results[metric_id]['response']}
                            ]
                        }
                    ])

def main():
    parser = argparse.ArgumentParser(description='Aggregate model assessments and generate final ground truth')
    parser.add_argument('--video_dir', default='../video_clips_6s_2',
                      help='Path to the folder containing video files (default: ../video_clips_6s_2)')
    parser.add_argument('--output_dir', type=str, default="5_output", help="Directory to save output files")
    parser.add_argument('--start_round', type=int, default=1, choices=[1, 2, 3],
                      help="Round to start processing from (1, 2, or 3)")
    parser.add_argument('--batch_num', type=int, default=5, 
                      help="Number of simultaneous batch processing")
    
    args = parser.parse_args()
    
    processor = ARVideoProcessor(output_dir=args.output_dir, batch_num=args.batch_num)
    processor.process_videos(args.video_dir, args.start_round)

if __name__ == "__main__":
    main()