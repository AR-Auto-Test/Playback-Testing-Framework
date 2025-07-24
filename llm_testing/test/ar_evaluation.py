"""
AR Evaluation Common Utilities

This module provides common utilities for AR evaluation across different models,
including video processing, JSON handling, and result formatting.
"""

import os
import re
import json
import cv2
import base64
import numpy as np
import time
import traceback
import datetime
import threading
from pathlib import Path
from decord import VideoReader, cpu
from PIL import Image
from typing import List, Dict, Any, Set, Tuple, Optional


def read_video_frames(video_path: str, interval: int = 10) -> List[str]:
    """Read video frames and convert to base64.
    
    Args:
        video_path: Path to the video file
        interval: Sampling interval for frames
        
    Returns:
        List of base64-encoded JPEG frames
    """
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        # Compress frame to reduce memory usage
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    
    # Sample frames at regular intervals
    output = base64Frames[0::interval]
    print(f"{len(base64Frames)} frames read from {video_path}, {len(output)} frames are sampled")
    
    return output


def load_video_decord(video_path: str, num_segments: int = 32) -> Tuple[List[Image.Image], List[int]]:
    """Load video frames using Decord library.
    
    Args:
        video_path: Path to the video file
        num_segments: Number of segments to divide the video into
        
    Returns:
        Tuple containing list of PIL images and list of frame indices
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    # Calculate frame indices to sample
    frame_indices = get_frame_indices(None, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    # Extract frames
    images = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        images.append(img)
    
    return images, frame_indices.tolist()


def get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get frame indices to sample from video.
    
    Args:
        bound: Optional tuple of (start, end) time in seconds
        fps: Frames per second of the video
        max_frame: Maximum frame index
        first_idx: First frame index to consider
        num_segments: Number of segments to divide the video into
        
    Returns:
        Numpy array of frame indices
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
        
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) 
        for idx in range(num_segments)
    ])
    
    return frame_indices


def extract_extension_json_content(text: str) -> dict:
    """Extract JSON content for extension question from model response."""
    # Try simple regex pattern first
    simple_pattern = r'{[\s\S]*?}'
    match = re.search(simple_pattern, text)
    
    if match:
        try:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            # Check if it has the expected fields for extension question
            if "Metrics" in parsed and "Issue Found" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, return None
    return None

def extract_json_content(text: str) -> str:
    """Extract JSON content from model response with improved pattern matching.
    
    Args:
        text: Text response from model
        
    Returns:
        JSON string if found, None otherwise
    """
    # First try the simple regex pattern
    simple_pattern = r'{[\s\S]*?}'
    match = re.search(simple_pattern, text)
    
    if match:
        try:
            # Verify it's valid JSON by parsing it
            json_str = match.group(0)
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If parsing fails, the match might be incomplete or incorrect
            pass
    
    # Try more advanced pattern matching - look for JSON with required fields
    patterns = [
        # Pattern with balanced braces
        r'{(?:[^{}]|(?R))*}',
        # Looser pattern, find anything between braces
        r'{\s*"[^"]+"\s*:.*?}',
        # Try to find JSON-like structure with the expected fields
        r'{(?:[^{}]|"[^"]*")*"Video_name"[^{}]*"Metrics"[^{}]*"Issue"[^{}]*"Reason"[^{}]*}'
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, re.DOTALL)
            for potential_json in matches:
                try:
                    parsed = json.loads(potential_json)
                    # Check if it has the expected fields
                    if all(key in parsed for key in ["Metrics", "Issue", "Reason"]):
                        return potential_json
                except json.JSONDecodeError:
                    continue
        except:
            continue
    
    # If we still can't find valid JSON, try a more manual approach
    try:
        # Find the beginning of a JSON object
        start_idx = text.find('{')
        if start_idx != -1:
            # Track brace balance to find the matching closing brace
            balance = 1
            for i in range(start_idx + 1, len(text)):
                if text[i] == '{':
                    balance += 1
                elif text[i] == '}':
                    balance -= 1
                    if balance == 0:
                        # Found the matching closing brace
                        potential_json = text[start_idx:i+1]
                        try:
                            json.loads(potential_json)
                            return potential_json
                        except json.JSONDecodeError:
                            # If it's still not valid, we can try to fix common issues
                            # Like missing quotes around keys, extra commas, etc.
                            pass
    except:
        pass
    
    # As a last resort, try to construct a valid JSON from the text
    if "Video_name" in text and "Metrics" in text and "Issue" in text and "Reason" in text:
        try:
            # Extract key information from the text
            video_name_match = re.search(r'"Video_name"\s*:\s*"([^"]+)"', text)
            metrics_match = re.search(r'"Metrics"\s*:\s*"([^"]+)"', text)
            issue_match = re.search(r'"Issue"\s*:\s*(true|false)', text)
            reason_match = re.search(r'"Reason"\s*:\s*"([^"]*(?:[^"]*"[^"]*"[^"]*)*)"', text)
            
            if video_name_match and metrics_match and issue_match and reason_match:
                constructed_json = {
                    "Video_name": video_name_match.group(1),
                    "Metrics": metrics_match.group(1),
                    "Issue": issue_match.group(1) == "true",
                    "Reason": reason_match.group(1)
                }
                return json.dumps(constructed_json)
        except:
            pass
    
    # If all attempts fail, return None
    return None
    

def save_json_results(file_path: str, video_file: str, evaluation_results: Dict, continue_file: str = None, file_lock: Optional[threading.Lock] = None) -> List[Dict]:
    """Save JSON results to file, ensuring existing data is preserved with thread-safe file operations.
    
    Args:
        file_path: Path to save the results
        video_file: Name of the video file
        evaluation_results: Dict of evaluation results keyed by metric
        continue_file: Optional path to file with existing results to continue from
        file_lock: Optional threading lock to use for file operations (should already be acquired by caller)
    
    Returns:
        List of all results after saving
    """
    def _save_json_results_internal():
        """Internal function that performs the actual save operation"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Initialize results array
        all_results = []
        
        # If in continue mode, try to load existing results from continue_file
        if continue_file and os.path.exists(continue_file):
            try:
                with open(continue_file, 'r') as f:
                    continue_results = json.load(f)
                    if isinstance(continue_results, list):
                        all_results = continue_results
                        print(f"Loaded {len(all_results)} existing results from continue file: {continue_file}")
                    else:
                        print(f"Warning: Continue file {continue_file} has unexpected format, expected a list")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load continue file {continue_file}: {str(e)}")
        
        # If current output file exists and is not empty, try to load existing results
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                with open(file_path, 'r') as f:
                    existing_results = json.load(f)
                    if isinstance(existing_results, list):
                        # If no results were loaded from continue_file, use current file's results
                        if not all_results:
                            all_results = existing_results
                            print(f"Loaded {len(all_results)} existing results from current file: {file_path}")
                        # Otherwise, ensure no results are lost when merging
                        else:
                            # Create a set of existing video+metric combinations to avoid duplicates
                            existing_entries = {(item.get('Video_name', ''), item.get('Metrics', '')) 
                                              for item in all_results}
                            
                            for item in existing_results:
                                entry_key = (item.get('Video_name', ''), item.get('Metrics', ''))
                                if entry_key not in existing_entries:
                                    all_results.append(item)
                                    existing_entries.add(entry_key)
                            
                            print(f"Merged results from current file. Total results: {len(all_results)}")
                    else:
                        print(f"Warning: Current file {file_path} has unexpected format, expected a list")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load current file {file_path}: {str(e)}")
        
        # Add new evaluation results
        for metric, result in evaluation_results.items():
            # Skip extension results (they're not saved to JSON)
            if metric == "Extension":
                continue
                
            # Ensure each result has the correct video name
            result['Video_name'] = video_file
            
            # Check if this result already exists
            exists = False
            for existing in all_results:
                if (existing.get('Video_name') == video_file and 
                    existing.get('Metrics') == result.get('Metrics')):
                    exists = True
                    break
            
            if not exists:
                all_results.append(result)
        
        # Write all results to file atomically
        # First write to a temporary file, then rename to avoid partial writes
        temp_file_path = file_path + '.tmp'
        try:
            with open(temp_file_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Atomic rename operation
            os.rename(temp_file_path, file_path)
            print(f"Saved {len(all_results)} total results to {file_path}")
            
        except Exception as e:
            # Clean up temporary file if something went wrong
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise e
        
        return all_results
    
    # 如果传入了外部锁，说明调用者已经获取了锁，直接执行保存操作
    if file_lock is not None:
        return _save_json_results_internal()
    else:
        # 如果没有传入外部锁，使用内部锁
        with threading.Lock():
            return _save_json_results_internal()


def get_diverse_params(sample_index: int, total_samples: int) -> Dict:
    """Generate diverse parameters for Best-of-N strategy.
    
    Args:
        sample_index: Index of the sample (0-based)
        total_samples: Total number of samples
        
    Returns:
        Dict of generation parameters
    """
    # Scale parameters based on sample index
    temperature = 0.5 + (0.5 * sample_index / total_samples)  # From 0.5 to 1.0
    top_p = 0.85 + (0.1 * sample_index / total_samples)       # From 0.85 to 0.95
    
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 40,
        "max_new_tokens": 2048
    }


def select_best_response(responses: List[str], json_data: List[Dict]) -> Tuple[str, Dict]:
    """Select best response from multiple samples using majority voting.
    
    Args:
        responses: List of text responses from model
        json_data: List of parsed JSON data from responses
        
    Returns:
        Tuple of (best response text, best JSON data)
    """
    # If no JSON data was successfully parsed, return the first response
    if not json_data:
        return responses[0], None
    
    # Count true/false values for "Issue" field
    issue_counts = {"true": 0, "false": 0}
    valid_responses = []
    valid_json = []
    
    for i, json_obj in enumerate(json_data):
        if "Issue" in json_obj:
            issue_value = str(json_obj["Issue"]).lower()
            issue_counts[issue_value] = issue_counts.get(issue_value, 0) + 1
            valid_responses.append(responses[i])
            valid_json.append(json_obj)
    
    # If no valid responses, return the first one
    if not valid_responses:
        return responses[0], None
    
    # Determine majority vote
    majority_issue = "true" if issue_counts.get("true", 0) > issue_counts.get("false", 0) else "false"
    
    # Find the first response that matches the majority vote
    for i, json_obj in enumerate(valid_json):
        if str(json_obj["Issue"]).lower() == majority_issue:
            return valid_responses[i], json_obj
    
    # Default to first valid response
    return valid_responses[0], valid_json[0]


def generate_timestamp_filename(base_filename: str) -> str:
    """Generate a filename with timestamp.
    
    Args:
        base_filename: Base filename to append timestamp to
        
    Returns:
        Filename with timestamp appended
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name, extension = os.path.splitext(base_filename)
    return f"{base_name}_{timestamp}{extension}"


def get_processed_videos(json_file: str) -> Set[str]:
    """Get set of videos that have been processed with ALL 6 metrics.
    
    Args:
        json_file: Path to JSON file with processed results
        
    Returns:
        Set of video filenames that have been processed with all 6 metrics
    """
    from ar_prompts import METRICS  # 导入所需的metrics列表
    
    processed = set()
    video_metrics = {}  # Track metrics per video
    
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                
            if isinstance(results, list):
                # Group metrics by video name
                for item in results:
                    if 'Video_name' in item and 'Metrics' in item:
                        video_name = item['Video_name']
                        metric = item['Metrics']
                        
                        if video_name not in video_metrics:
                            video_metrics[video_name] = set()
                        video_metrics[video_name].add(metric)
                
                # Only include videos that have ALL required metrics
                for video_name, metrics in video_metrics.items():
                    if len(metrics) >= len(METRICS):  # Has all 6 metrics
                        # Double check that all required metrics are present
                        if all(required_metric in metrics for required_metric in METRICS):
                            processed.add(video_name)
                        else:
                            missing = set(METRICS) - metrics
                            print(f"Video {video_name} missing metrics: {missing}")
                    else:
                        print(f"Video {video_name} has only {len(metrics)}/{len(METRICS)} metrics")
                        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load processed videos from {json_file}: {str(e)}")
    
    return processed

def format_time(seconds):
    """将秒数格式化为更易读的形式（时:分:秒）。
    
    Args:
        seconds: 需要格式化的秒数
        
    Returns:
        格式化后的时间字符串，格式为 HH:MM:SS
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"