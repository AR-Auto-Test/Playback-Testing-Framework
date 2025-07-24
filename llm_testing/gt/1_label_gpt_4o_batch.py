"""
python 1_label_gpt_4o_batch.py --mode discard --batch_size 3 --video_dir ../video_clips_6s_2_test (to start fresh and process 5 videos at once)
python batch_gpt_4o.py --mode continue --batch_size 3 (to resume from where it left off with 3 videos at once)
"""

import cv2
import base64
import time
import json
import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
import traceback
import argparse
import re
import uuid
from pathlib import Path

load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

__INTERVAL = 10

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

def generate_system_prompt():
    return """You are a specialized AI assistant trained to review AR applications based on screen recordings. The video provided is a screen recording from a mobile phone that demonstrates an AR (Augmented Reality) application. We specifically designed and selected certain environment and use cases to test the AR application to identify any potential performance issues with AR functionality. In all the screen recording videos, the movement of AR objects is simulated programmatically as interactions on the phone screen. Currently, the simulated interactions include the following two types: 1.Tapping the screen: The AR object will immediately appear at the corresponding position. 2. Swiping the screen: The AR object will move along the swipe path to the corresponding positions.

Your task is to evaluate the realism of AR effect and the performance of AR apps based on following metrics: 1. Object Placement. 2. Object Movement. 3. Occlusion. 4. Lighting. 5. Visual Artifacts and Rendering Issues. 6. Black Screen 

And you could ignore the following aspects in evaluation: 
1. The size of AR object. Because some apps may privide the resize feature so that the user could adjust. 
2. The stylish or cartoonish AR object. Because some apps may intentionally provide stylish and cartoonish models according to the theme of app. The The stylish or cartoonish does not account for realism issue.
3. The UI elements. The effect and visibility of UI elements like grid lines or measures are part of design by developers. Therefore, the interaction between AR object and UI element is out of the scope of evaluation.

For evaluation questions, I will ask you to response in JSON format. Please ensure your response is a valid JSON object with this format strictly. Do not include any text outside the JSON object. Ensure all keys are enclosed in double quotes and use proper JSON syntax. The response should match the example below:
{
    "Video_name": string,   
    "Metrics": string, // the metric to evaluate, e.g. Object Placement
    "Issue": boolean, // true if there are issues found 
    "Reason":  string //The reason why you think there are issues or no issues. Provide explanation.
}"""

def process_batch(video_paths, output_file, json_output_file, max_wait_time=15*60, max_retries=3):
    """Process a batch of videos with GPT-4o using OpenAI's Batch API with multi-turn conversation."""
    try:
        # Read frames for all videos in the batch
        videos_data = {}
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            print(f"Reading frames for {video_name}")
            frames = read_video_frames(video_path, __INTERVAL)
            videos_data[video_name] = {
                "frames": frames,
                "path": video_path,
                "conversation_history": [],
                "log": [f"Processing video: {video_path}\n\n"],
                "json_results": []
            }
        
        # Initial questions
        initial_questions = [
            "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video?",
            "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it."
        ]
        
        # Process initial questions in a multi-turn conversation
        for question_idx, question in enumerate(initial_questions):
            print(f"Processing initial question {question_idx + 1} for {len(videos_data)} videos")
            
            # Create batch input file for this question
            batch_input_path = create_batch_conversation_file(videos_data, question, question_idx)
            
            try:
                # Process batch
                process_batch_request(
                    batch_input_path, 
                    videos_data, 
                    question, 
                    question_idx,
                    max_wait_time=max_wait_time,
                    max_retries=max_retries
                )
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"Error processing initial question {question_idx + 1}: {str(e)}")
                print(f"Traceback: {error_traceback}")
                
                # Log the detailed error for each video
                for video_name in videos_data:
                    error_message = f"Error: {str(e)}\n\nTraceback:\n{error_traceback}"
                    videos_data[video_name]["log"].append(f"Question: {question}\nAnswer: {error_message}\n{'-' * 20}\n")
        
        # Process metrics with JSON responses
        metrics = [
            "Object Placement",
            "Object Movement",
            "Occlusion", 
            "Lighting",
            "Visual Artifacts and Rendering Issues",
            "Black Screen"
        ]
        
        for metric_idx, metric in enumerate(metrics):
            question = f"Is there any issue with the {metric}? Please clearly state your reason and give your answer in JSON format."
            print(f"Processing metric '{metric}' for {len(videos_data)} videos")
            
            # Create batch input file for this metric
            batch_input_path = create_batch_conversation_file(videos_data, question, metric_idx + len(initial_questions))
            
            try:
                # Process batch
                process_batch_request(batch_input_path, videos_data, question, metric_idx + len(initial_questions), metric=metric, max_wait_time=max_wait_time, max_retries=max_retries)
            except Exception as e:
                error_traceback = traceback.format_exc()
                print(f"Error processing metric '{metric}': {str(e)}")
                print(f"Traceback: {error_traceback}")
                
                # Log the detailed error for each video
                for video_name in videos_data:
                    error_message = f"Error: {str(e)}\n\nTraceback:\n{error_traceback}"
                    videos_data[video_name]["log"].append(f"Question: {question}\nAnswer: {error_message}\n{'-' * 20}\n")
        
        # Write results to files
        with open(output_file, "a") as f:
            for video_name, data in videos_data.items():
                f.writelines(data["log"])
                f.write("\n")

        with open(json_output_file, "a") as f:
            for video_name, data in videos_data.items():
                for json_item in data["json_results"]:
                    json.dump(json_item, f, indent=4)
                    f.write("\n")
                    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error processing batch: {e}")
        print(f"Traceback: {error_traceback}")
        
        with open(output_file, "a") as f:
            f.write(f"Error processing batch: {str(e)}\n\nTraceback:\n{error_traceback}\n\n")

def create_batch_conversation_file(videos_data, question, turn_idx):
    """Create a JSONL batch input file for multiple videos with conversation history."""
    # Create a temporary file to store the JSONL data
    temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
    file_path = temp_file.name
    temp_file.close()
    
    with open(file_path, 'w') as f:
        for video_name, data in videos_data.items():
            frames = data["frames"]
            conversation_history = data["conversation_history"]
            
            # Convert frames to OpenAI content format
            frame_images = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "low"}} for frame in frames]
            
            # Prepare the initial content with video frames
            initial_content = [
                {"type": "text", "text": f"The video name is {video_name}. These are frames from a video that I want to analyze."}
            ] + frame_images
            
            # Create messages array with conversation history
            messages = [
                {"role": "system", "content": generate_system_prompt()}
            ]
            
            # For the first turn, include frames in the user message
            if turn_idx == 0:
                messages.append({
                    "role": "user", 
                    "content": initial_content + [{"type": "text", "text": question}]
                })
            else:
                # For first turn, conversation_history will be empty
                # Add all previous turns to the conversation
                for prev_turn in conversation_history:
                    messages.extend(prev_turn)
                
                # Add current question
                messages.append({
                    "role": "user", 
                    "content": question
                })
            
            # Create the batch request with a custom ID format that's easy to parse later
            # Use a separator that won't appear in video names (:::) between video name and turn index
            batch_request = {
                "custom_id": f"{video_name}:::{turn_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": messages,
                    "temperature": 1,
                    "max_tokens": 2048
                }
            }
            
            # Write to the JSONL file
            f.write(json.dumps(batch_request) + '\n')
    
    return file_path

def process_batch_request(batch_input_path, videos_data, question, turn_idx, metric=None, 
                      max_wait_time=15*60, max_retries=3):
    """Process a batch request and update videos_data with responses."""
    retry_count = 0
    batch_file_id = None
    
    while retry_count < max_retries:
        try:
            # Upload the file to OpenAI (only on first attempt or if previous attempt failed)
            if batch_file_id is None:
                with open(batch_input_path, 'rb') as f:
                    file = client.files.create(
                        file=f,
                        purpose="batch"
                    )
                batch_file_id = file.id
            
            # Create a batch with the uploaded file
            batch = client.batches.create(
                input_file_id=batch_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Set batch start time
            batch_start_time = time.time()
            
            # Poll until the batch is complete or timeout is reached
            while True:
                # Check if timeout has been reached
                current_time = time.time()
                elapsed_time = current_time - batch_start_time
                
                if elapsed_time > max_wait_time:
                    print(f"Batch {batch.id} timed out after {max_wait_time/60:.1f} minutes")
                    
                    # Try to cancel the batch
                    try:
                        client.batches.cancel(batch.id)
                        print(f"Cancelled batch {batch.id} due to timeout")
                    except Exception as cancel_error:
                        print(f"Failed to cancel batch {batch.id}: {cancel_error}")
                    
                    # Increment retry counter and retry
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying batch (attempt {retry_count+1}/{max_retries})...")
                        break  # Break the polling loop to retry
                    else:
                        raise TimeoutError(f"Batch processing timed out after {retry_count} retries")
                
                status = client.batches.retrieve(batch.id)
                
                # 处理完成或失败的情况
                if status.status == "completed":
                    # 处理成功的响应...
                    return  # 成功完成
                
                if status.status in ["failed", "expired", "cancelled"]:
                    # Increment retry counter
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Batch failed. Retrying (attempt {retry_count+1}/{max_retries})...")
                        break  # Break the polling loop to retry
                    else:
                        raise Exception(f"Batch failed with status: {status.status}")
                
                print(f"Waiting for batch completion...")
                time.sleep(60)  # 每分钟检查一次
        
        except Exception as e:
            # 异常处理和重试逻辑
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying after error (attempt {retry_count+1}/{max_retries})...")
                time.sleep(60)  # 等待后重试
            else:
                # 清理并重新抛出异常
                if os.path.exists(batch_input_path):
                    os.remove(batch_input_path)
                raise

def main():
    video_dir = "../video_clips_6s_2"
    output_file = "log/result_batch_gpt4o.txt"
    json_output_file = "json/label_batch_gpt4o.json"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process videos with GPT-4o using OpenAI's Batch API.")
    parser.add_argument(
        "--mode", choices=["discard", "continue"], default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos."
    )
    parser.add_argument(
        "--batch_size", type=int, default=5,
        help="Number of videos to process in a single batch (default: 5)"
    )
    parser.add_argument(
        "--video_dir", type=str, default="../video_clips_6s_2",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--max_wait_time", type=int, default=15*60,
        help="Maximum wait time in seconds for a batch to complete before retrying (default: 900 seconds / 15 minutes)"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Maximum number of retry attempts for a failed batch (default: 3)"
    )
    args = parser.parse_args()
    
    # Update video directory if specified
    video_dir = args.video_dir

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    os.makedirs("debug", exist_ok=True)  # Create debug directory

    processed_videos = set()
    if args.mode == "discard":
        # Clear output files
        with open(output_file, 'w') as f:
            pass
        with open(json_output_file, 'w') as f:
            pass
    else:
        # Parse log file to track processed videos
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    if line.startswith("Processing video:"):
                        video_name = os.path.basename(line.split("Processing video:")[1].strip())
                        processed_videos.add(video_name)

    # Get all video files
    video_files = [
        os.path.join(video_dir, filename)
        for filename in os.listdir(video_dir)
        if filename.endswith((".mp4", ".mov"))
    ]

    # Filter out already processed videos in 'continue' mode
    if args.mode == "continue":
        video_files = [vf for vf in video_files if os.path.basename(vf) not in processed_videos]

    if not video_files:
        print("No videos to process.")
        return

    total_videos = len(video_files)
    print(f"Found {len(processed_videos)} videos processed. {total_videos} videos to process.")

    # Process videos in batches
    for i in range(0, total_videos, args.batch_size):
        batch = video_files[i:i+args.batch_size]
        print(f"Processing batch {i//args.batch_size + 1}/{(total_videos + args.batch_size - 1)//args.batch_size}: {len(batch)} videos")
        process_batch(batch, output_file, json_output_file, max_wait_time=args.max_wait_time, max_retries=args.max_retries)
        print(f"Completed batch {i//args.batch_size + 1}")

if __name__ == "__main__":
    main()