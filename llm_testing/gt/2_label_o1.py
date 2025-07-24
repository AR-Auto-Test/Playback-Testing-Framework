"""
python 2_label_o1.py --mode discard (to start fresh)
python 2_label_o1.py --mode continue (to resume from where it left off)
python 2_label_o1.py --mode update --update_metric "Lighting" (to only update a specific metric)
"""

import cv2
import base64
import time
from openai import OpenAI
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback
import argparse
import json
import re

# Import the new prompts module
from ar_prompts_labels import generate_system_prompt, generate_ar_metrics_description, get_metrics, METRICS

load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

__INTERVAL = 10
MAX_WORKERS = 4  # 设置最大并发数
file_lock = Lock()  # 用于保护文件写操作的锁

def read_video_frames(video_path, interval=5):
    """Read video frames and convert to base64."""
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    
    output = base64Frames[0::interval]
    print(f"{len(base64Frames)} frames read from {video_path}, {len(output)} frames are sampled")
    
    return base64Frames[0::interval]

def validate_metric(metric_name):
    """Validate if the provided metric name is valid."""
    valid_metrics = METRICS
    if metric_name not in valid_metrics:
        valid_metrics_str = ", ".join(f"'{m}'" for m in valid_metrics)
        raise ValueError(f"Invalid metric name: '{metric_name}'. Valid metrics are: {valid_metrics_str}")
    return True

def load_processed_videos(json_output_file):
    """Load list of already processed videos and their processed metrics from the JSON file."""
    video_metrics = {}  # Dictionary to store video_name -> set of processed metrics
    total_metrics_processed = 0
    
    # Check if JSON file exists
    if os.path.exists(json_output_file):
        try:
            # Read the entire file content
            with open(json_output_file, 'r') as f:
                content = f.read()
            
            # Split the content by closing and opening braces to identify complete JSON objects
            json_objects = []
            brace_count = 0
            start_index = 0
            
            for i, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_index = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Extract complete JSON object
                        json_str = content[start_index:i+1]
                        json_objects.append(json_str)
            
            # Process each JSON object
            for json_str in json_objects:
                try:
                    json_data = json.loads(json_str)
                    if 'Video_name' in json_data and 'Metrics' in json_data:
                        video_name = json_data['Video_name']
                        metric = json_data['Metrics']
                        
                        if video_name not in video_metrics:
                            video_metrics[video_name] = set()
                        
                        video_metrics[video_name].add(metric)
                        total_metrics_processed += 1
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON object: {e}")
        
        except Exception as e:
            print(f"Error reading {json_output_file}: {e}")
    
    print(f"Found {len(video_metrics)} videos with {total_metrics_processed} metrics already processed")
    
    return video_metrics

def backup_json_file(json_output_file):
    """Create a backup of the JSON file before making updates."""
    if os.path.exists(json_output_file):
        backup_file = f"{json_output_file}.bak.{int(time.time())}"
        import shutil
        shutil.copy2(json_output_file, backup_file)
        print(f"Created backup of JSON file at: {backup_file}")
    return

def filter_json_file(json_output_file, metric_to_update):
    """Filter out entries for the specified metric from the JSON file."""
    if not os.path.exists(json_output_file):
        return
    
    # Read all lines from the file
    with open(json_output_file, 'r') as f:
        lines = f.readlines()
    
    # Filter out lines containing the specified metric
    filtered_lines = []
    for line in lines:
        try:
            json_data = json.loads(line.strip())
            if 'Metrics' in json_data and json_data['Metrics'] == metric_to_update:
                # Skip this line (it's for the metric we want to update)
                continue
            filtered_lines.append(line)
        except json.JSONDecodeError:
            # Keep invalid JSON lines (unlikely, but just in case)
            filtered_lines.append(line)
    
    # Write filtered lines back to the file
    with open(json_output_file, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"Filtered out entries for metric '{metric_to_update}' from the JSON file")

def process_video(video_path, output_file, json_output_file, processed_metrics=None, update_metric=None):
    """Processes a single video with GPT-4 Vision and logs both complete responses and JSON data."""
    try:
        base64Frames = read_video_frames(video_path, __INTERVAL)
        frame_images = list(map(lambda x: {"image": x}, base64Frames))
        video_name = video_path.split('/')[-1]

        # Get the set of metrics already processed for this video
        metrics_to_skip = set()
        if processed_metrics and video_name in processed_metrics:
            metrics_to_skip = processed_metrics[video_name]
            if metrics_to_skip and update_metric is None:
                print(f"Skipping previously processed metrics for {video_name}: {', '.join(metrics_to_skip)}")
                
        # If we're updating a specific metric, ensure it's not in the skip list
        if update_metric:
            if update_metric in metrics_to_skip:
                metrics_to_skip.remove(update_metric)
                print(f"Will re-evaluate metric '{update_metric}' for {video_name}")

        # Use new system prompt from ar_prompts_labels
        system_message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": generate_system_prompt()
                    }
                ]
            }
        ]

        # Step 1: Initial conversation to establish context
        initial_conversation = system_message + [
            {
                "role": "user", 
                "content": [
                    f"The video name is {video_name}. These are frames from a video that I want to upload. Please check them and I will ask questions later.",
                    *frame_images
                ]
            },
            {
                "role": "user",
                "content": "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video? Can you describe the setting (indoor/outdoor), the primary surfaces or objects in view, and any notable lighting conditions?"
            }
        ]
        
        params = {
            "model": "o3",
            "messages": initial_conversation,
            "temperature": 1,
            "max_completion_tokens": 2048,
        }

        response = client.chat.completions.create(**params)
        answer1 = response.choices[0].message.content
        
        # Step 2: Get 3D object description
        initial_conversation.append({"role": "assistant", "content": answer1})
        initial_conversation.append({
            "role": "user",
            "content": "What is the 3D object in the video? If you can see the 3D object, describe it in detail including its type, color, and any notable features. If you are not clear what exactly the object is, try to describe it as specifically as possible."
        })
        
        response = client.chat.completions.create(**params)
        answer2 = response.choices[0].message.content
        
        initial_conversation.append({"role": "assistant", "content": answer2})
        
        result_log = [
            f"Processing video: {video_path}\n\n",
            f"Question: What do you see in this video?\nAnswer: {answer1}\n{'-' * 20}\n",
            f"Question: What is the 3D object in the video?\nAnswer: {answer2}\n{'-' * 20}\n"
        ]
        
        # Step 3: Evaluate all AR aspects using metrics from ar_prompts_labels
        metrics = METRICS
        
        json_results = []
        
        for i, metric in enumerate(metrics):
            # If update_metric is specified, only process that metric
            if update_metric and metric != update_metric:
                continue
                
            # Skip already processed metrics (unless we're specifically updating this one)
            if metric in metrics_to_skip and (update_metric is None or metric != update_metric):
                print(f"  Skipping already processed metric: {metric}")
                continue
                
            print(f"  Processing metric: {metric}")
            
            # Get detailed description for the metric
            metric_description = generate_ar_metrics_description(i)
            question = f"Is there any issue with the {metric}? \n{metric_description}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'."
            
            single_conversation = initial_conversation + [
                {"role": "user", "content": question}
            ]
            
            params["messages"] = single_conversation
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                response = client.chat.completions.create(**params)
                answer = response.choices[0].message.content
                
                # Save complete response to log
                result_log.append(f"Question: {question}\nAnswer: {answer}\n{'-' * 20}\n")
                
                # Try to extract JSON from response
                try:
                    json_match = re.search(r"\{[^{}]*\}", answer)
                    if json_match:
                        json_str = json_match.group(0)
                        json_data = json.loads(json_str)
                        
                        # Ensure the correct video name is in the JSON
                        json_data['Video_name'] = video_name
                        
                        # Ensure the correct metric is in the JSON
                        if 'Metrics' not in json_data or json_data['Metrics'] != metric:
                            json_data['Metrics'] = metric
                            
                        # Ensure all required fields are present
                        if all(key in json_data for key in ["Video_name", "Metrics", "Issue", "Reason"]):
                            json_results.append(json_data)
                            break
                    retry_count += 1
                    time.sleep(2)  # Wait before retrying
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing JSON for {metric}: {str(e)}")
                    retry_count += 1
                    time.sleep(2)
            
            if retry_count == max_retries:
                print(f"Failed to get valid JSON response for {metric} after {max_retries} attempts")
                # Add a placeholder JSON result with error indication
                json_results.append({
                    "Video_name": video_name,
                    "Metrics": metric,
                    "Issue": False,
                    "Reason": "Could not determine from model output (JSON parsing failed)",
                    "error": "Failed to parse valid JSON after multiple attempts"
                })
        
        # Write complete responses to log file
        with file_lock:
            with open(output_file, "a") as f:
                f.writelines(result_log)
                f.write("\n")
        
        # Write JSON results to JSON file
        if json_results:
            with file_lock:
                with open(json_output_file, "a") as f:
                    for result in json_results:
                        json.dump(result, f, indent=4)
                        f.write("\n")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        traceback.print_exc()
        with file_lock:
            with open(output_file, "a") as f:
                f.write(f"Error processing video: {video_path}. Error: {e}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Process videos with o3 Vision.")
    parser.add_argument(
        "--mode", choices=["discard", "continue", "update"], default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos, 'update' to update specific metrics."
    )
    parser.add_argument(
        "--update_metric", type=str,
        help="Specify which metric to update (only used with --mode update)."
    )
    parser.add_argument(
        "--video_dir", type=str, default="../video_clips_6s_2",
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--output_file", type=str, default="log/result_o3.txt",
        help="File to save complete conversation logs"
    )
    parser.add_argument(
        "--json_output", type=str, default="json/label_o3.json",
        help="JSON file to save evaluation metrics"
    )
    
    args = parser.parse_args()
    
    # Update paths from args
    video_dir = args.video_dir
    output_file = args.output_file
    json_output_file = args.json_output
    
    # Handle update mode
    update_metric = None
    if args.mode == "update":
        if not args.update_metric:
            print("Error: --update_metric must be specified when using --mode update")
            return
        update_metric = args.update_metric
        try:
            validate_metric(update_metric)
            print(f"Update mode: Will only process metric '{update_metric}'")
        except ValueError as e:
            print(f"Error: {e}")
            return

    processed_videos = set()
    processed_metrics = {}
    
    if args.mode == "discard":
        # Clear output files
        with open(output_file, 'w') as f:
            pass
        with open(json_output_file, 'w') as f:
            pass
    elif args.mode == "update":
        # Create backup of existing results
        backup_json_file(json_output_file)
        # Load existing metrics
        processed_metrics = load_processed_videos(json_output_file)
        # Filter out entries for the metric we want to update
        filter_json_file(json_output_file, update_metric)
        # Create a header in the log file to indicate this is an update run
        with open(output_file, 'a') as f:
            from datetime import datetime
            f.write(f"\n{'-' * 80}\n")
            f.write(f"UPDATE RUN FOR METRIC: {update_metric} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'-' * 80}\n\n")
    else:  # continue mode
        # Parse log file to track processed videos
        if os.path.exists(json_output_file):
            processed_metrics = load_processed_videos(json_output_file)
            # For backward compatibility, also track fully processed videos
            for video_name, metrics in processed_metrics.items():
                if len(metrics) == len(METRICS):
                    processed_videos.add(video_name)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)

    video_files = [
        os.path.join(video_dir, filename)
        for filename in os.listdir(video_dir)
        if filename.endswith((".mp4", ".mov"))
    ]

    # Filter videos based on mode
    videos_to_process = []
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        
        # For update mode, include all videos
        if args.mode == "update":
            videos_to_process.append(video_path)
            continue
            
        # For continue mode, skip fully processed videos
        if args.mode == "continue" and video_name in processed_videos:
            continue
            
        videos_to_process.append(video_path)

    if not videos_to_process:
        print("No videos to process.")
        return

    if args.mode == "update":
        print(f"Update mode: Will update metric '{update_metric}' for {len(videos_to_process)} videos.")
    else:
        print(f"Found {len(processed_videos)} videos processed. Total videos to process: {len(videos_to_process)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_video, video_path, output_file, json_output_file, 
                                   processed_metrics=processed_metrics,
                                   update_metric=update_metric): video_path for video_path in videos_to_process}
        
        for i, future in enumerate(as_completed(futures), start=1):
            video_path = futures[future]
            try:
                future.result()
                if args.mode == "update":
                    print(f"Finished updating '{update_metric}' for {video_path} ({i}/{len(videos_to_process)})")
                else:
                    print(f"Finished processing {video_path} ({i}/{len(videos_to_process)})")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()