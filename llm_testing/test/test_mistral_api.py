"""
python test_mistral_api.py --mode discard (to start fresh)
python test_mistral_api.py --mode continue (to resume from where it left off)

Additional options:
--fewshot True/False  (enable/disable few-shot learning, default is False)
--num_frames 8  (number of frames to sample from video, default is 8)
--video_dir path/to/videos  (directory containing videos to process)
--bon_samples 5  (number of samples for Best-of-N strategy, default is 5)
--continue_file
"""

import cv2
import base64
import time
import os
import torch
import numpy as np
import argparse
import json
import re
import traceback
import datetime
from mistralai import Mistral
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path

# Import the custom modules
from ar_prompts import (
    generate_system_prompt, 
    generate_few_shot_prompt, 
    generate_conversation_questions,
    generate_ar_metrics_description,
    METRICS
)
from ar_evaluation import (
    extract_json_content,  # Improved JSON extraction
    extract_extension_json_content,
    save_json_results,
    get_diverse_params,    # Parameter diversity for Best-of-N
    select_best_response,  # Best-of-N selection logic
    generate_timestamp_filename,
    get_processed_videos,
    format_time
)

load_dotenv()  # Load environment variables from .env file

file_lock = Lock()  # Thread lock for file operations
MAX_WORKERS = 8      # Maximum number of concurrent workers
MAX_API_FRAMES = 8   # Maximum frames that Mistral API can handle in one prompt
MAX_RETRIES = 3      # Maximum retry attempts for API errors
RETRY_DELAY = 5      # Initial retry delay in seconds

def sample_frames_evenly(video_path, num_frames=8):
    """Sample frames evenly from the video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        raise ValueError(f"Cannot read frames from video: {video_path}")
    
    # Calculate indices of frames to sample
    if num_frames > frame_count:
        num_frames = frame_count
    
    # Generate evenly spaced frame indices
    if num_frames == 1:
        indices = [frame_count // 2]  # Middle frame
    else:
        indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    base64Frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    cap.release()
    print(f"Successfully sampled {len(base64Frames)} frames from {video_path}")
    
    return base64Frames

def get_few_shot_examples(few_shot_dir, max_examples=2):
    """Get few-shot examples if available."""
    examples = []
    if not os.path.exists(few_shot_dir):
        print(f"Few-shot examples directory {few_shot_dir} does not exist")
        return examples
    
    image_files = [f for f in os.listdir(few_shot_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {few_shot_dir}")
        return examples
    
    # Limit number of examples to prevent exceeding API frame limit
    selected_files = image_files[:min(max_examples, len(image_files))]
    
    for image_file in selected_files:
        image_path = os.path.join(few_shot_dir, image_file)
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode("utf-8")
                examples.append({
                    "name": image_file,
                    "data": base64_image
                })
        except Exception as e:
            print(f"Error loading few-shot example {image_file}: {e}")
    
    print(f"Loaded {len(examples)} few-shot examples")
    return examples

def run_mistral_api_with_retry(client, model, messages, max_retries=MAX_RETRIES, initial_delay=RETRY_DELAY):
    """Run the Mistral API with exponential backoff retry logic."""
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            response = client.chat.complete(
                model=model,
                messages=messages
            )
            return response
        except Exception as e:
            last_error = e
            retry_count += 1
            
            if retry_count < max_retries:
                # Exponential backoff with jitter
                delay = initial_delay * (2 ** (retry_count - 1))
                jitter = delay * 0.1 * np.random.random()
                wait_time = delay + jitter
                
                print(f"API call failed (attempt {retry_count}/{max_retries}). "
                      f"Error: {str(e)}. Retrying in {wait_time:.2f} seconds...")
                
                time.sleep(wait_time)
            else:
                print(f"API call failed after {max_retries} attempts. Last error: {str(e)}")
                raise last_error

def process_video(video_path, output_file, json_output_file, few_shot_dir=None, use_few_shot=False, 
                  num_frames=8, bon_samples=5, continue_file=None):
    """
    Process a single video with the Mistral API using Best-of-N strategy for improved response quality.
    
    Args:
        video_path: Path to the video file
        output_file: Path to log file for complete responses
        json_output_file: Path to output file for JSON results
        few_shot_dir: Directory containing few-shot examples
        use_few_shot: Whether to use few-shot learning
        num_frames: Number of frames to sample from the video
        bon_samples: Number of samples for Best-of-N strategy
        continue_file: Path to file with existing results to continue from
    """
    try:
        # Initialize the Mistral client
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        
        # Specify model - use the multimodal model
        model = "mistral-small-latest"  # or "mistral-large" depending on your needs
        
        # Sample frames evenly from the video
        video_frames = sample_frames_evenly(video_path, num_frames=min(num_frames, MAX_API_FRAMES))
        video_name = os.path.basename(video_path)

        # System prompt from ar_prompts module
        system_prompt = generate_system_prompt()

        # Prepare content with images
        content = [{"type": "text", "text": f"The video name is {video_name}. These are frames from a video that I want to analyze."}]
        
        # Add few-shot examples if enabled
        few_shot_examples = []
        if use_few_shot and few_shot_dir:
            few_shot_examples = get_few_shot_examples(few_shot_dir, max_examples=MAX_API_FRAMES - len(video_frames))
            
            # Add few-shot examples to content
            for i, example in enumerate(few_shot_examples):
                content.append({"type": "text", "text": f"Few-shot example {i+1}:"})
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{example['data']}"
                })
        
        # Add video frames to content
        for frame in video_frames:
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{frame}"
            })
        
        # Add the initial question
        content.append({
            "type": "text",
            "text": "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video? Can you describe the setting (indoor/outdoor), the primary surfaces or objects in view, and any notable lighting conditions?"
        })
        
        # Create the message for the chat
        messages = [
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        # Get the chat response with retry logic
        response = run_mistral_api_with_retry(client, model, messages)
        answer1 = response.choices[0].message.content
        
        # Step 2: Get 3D object description - only send text now that model has seen the images
        messages.append({"role": "assistant", "content": answer1})
        messages.append({
            "role": "user",
            "content": "What is the 3D object in the video? If you can see the 3D object, describe it in detail including its type, color, and any notable features. If you are not clear what exactly the object is, try to describe it as specifically as possible."
        })
        
        response = run_mistral_api_with_retry(client, model, messages)
        answer2 = response.choices[0].message.content
        
        messages.append({"role": "assistant", "content": answer2})
        
        result_log = [
            f"Processing video: {video_path}\n\n",
            f"Question: What do you see in this video?\nAnswer: {answer1}\n{'-' * 20}\n",
            f"Question: What is the 3D object in the video?\nAnswer: {answer2}\n{'-' * 20}\n"
        ]
        
        # Step 3: Evaluate all AR aspects using Best-of-N strategy
        metrics = METRICS
        evaluation_results = {}
        metric_best_responses = {}
        
        # messages 已经包含了前2个验证问题的问答，直接作为基础对话历史
        base_messages = messages.copy()
        
        for i, metric in enumerate(metrics):
            # 每个metric都从相同的base_messages开始，确保独立评估
            
            # Get detailed description for the metric
            metric_description = generate_ar_metrics_description(i)
            question = f"Is there any issue with the {metric}? \n{metric_description}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'."
            
            # Best-of-N strategy: generate multiple responses with diverse parameters
            bon_responses = []
            bon_json_data = []
            
            # Generate multiple samples with diverse parameters
            for n in range(bon_samples):
                try:
                    # 使用独立的messages历史，不包含其他metrics的回答
                    sample_messages = base_messages.copy()
                    sample_messages.append({"role": "user", "content": question})
                    
                    # Get diverse parameters for this sample (注意：当前Mistral API调用没有使用这些参数)
                    params = get_diverse_params(n, bon_samples)
                    print(f"Generating sample {n+1}/{bon_samples} for metric {metric}")
                    
                    # Make API call with retry logic
                    response = run_mistral_api_with_retry(client, model, sample_messages)
                    answer = response.choices[0].message.content
                    bon_responses.append(answer)
                    
                    # Try to extract JSON from response
                    json_content = extract_json_content(answer)
                    if json_content:
                        try:
                            json_data = json.loads(json_content)
                            # Ensure correct video name and metric
                            json_data['Video_name'] = video_name
                            if 'Metrics' not in json_data:
                                json_data['Metrics'] = metric
                            bon_json_data.append(json_data)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON for sample {n+1} of metric {metric}: {str(e)}")
                    
                except Exception as e:
                    print(f"Error generating sample {n+1} for metric {metric}: {str(e)}")
                    # Continue with next sample even if this one fails
            
            # Select best response using majority voting
            best_response, best_json = select_best_response(bon_responses, bon_json_data)
            
            # 保存最佳回答用于拓展问题
            metric_best_responses[metric] = best_response
            
            # Add to result log
            result_log.append(f"Question: Is there any issue with the {metric}?\nAnswer: {best_response}\n{'-' * 20}\n")
            
            # Store JSON result
            if best_json:
                evaluation_results[metric] = best_json
            else:
                # Create default result if no valid JSON was found
                evaluation_results[metric] = {
                    "Video_name": video_name,
                    "Metrics": metric,
                    "Issue": False,  # Default to no issue
                    "Reason": "Could not determine from model output",
                    "error": "No valid JSON found"
                }
        
        # Step 4: Extension question (基于所有6个metrics的完整评估)
        extension_question = ("In addition to the six metrics we've already evaluated (Object Placement, Object Movement, "
                            "Occlusion, Lighting, Visual Artifacts and Rendering Issues, and Black Screen), please carefully "
                            "examine this AR video for any other specific issues or problems that you can observe. Focus on "
                            "describing actual problems you see in the video rather than listing potential metrics. Please provide "
                            "your answer in JSON format with the following fields: 'Metrics' (indicating the category or aspect "
                            "of the issue), 'Issue Found' (describing the specific problem you observed in this video).")
        
        try:
            # 构建包含所有6个metrics评估结果的messages
            extension_messages = base_messages.copy()
            
            # 添加所有6个metrics的实际问答对
            for i, metric in enumerate(metrics):
                metric_description = generate_ar_metrics_description(i)
                metric_question = f"Is there any issue with the {metric}? \n{metric_description}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'."
                
                extension_messages.append({"role": "user", "content": metric_question})
                extension_messages.append({"role": "assistant", "content": metric_best_responses[metric]})
            
            # 添加拓展问题
            extension_messages.append({"role": "user", "content": extension_question})
            
            # 基于完整的评估历史提出拓展问题
            response = run_mistral_api_with_retry(client, model, extension_messages)
            extension_answer = response.choices[0].message.content
            
            # Try to extract and display JSON structure for easy viewing
            extension_json = extract_extension_json_content(extension_answer)
            #if extension_json:
            #    print(f"Extension findings for {video_name}:")
            #    print(f"  Metrics: {extension_json.get('Metrics', 'N/A')}")
            #    print(f"  Issue Found: {extension_json.get('Issue Found', 'N/A')}")
            
            # Add to result log only (not saved to JSON output)
            result_log.append(f"Extension Question: Other AR issues?\nAnswer: {extension_answer}\n{'-' * 20}\n")
                
        except Exception as e:
            print(f"Error processing extension question for {video_name}: {str(e)}")
            result_log.append(f"Extension Question: Other AR issues?\nAnswer: Error - {str(e)}\n{'-' * 20}\n")
        
        # Write complete responses to log file
        with file_lock:
            # 写入完整对话日志
            with open(output_file, "a") as f:
                f.writelines(result_log)
                f.write("\n")
            
            # 保存JSON结果 (传入file_lock参数)
            save_json_results(
                json_output_file,
                video_name,
                evaluation_results,
                continue_file=continue_file,
                file_lock=file_lock  # 传入现有的锁
            )
        
        return True

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        traceback.print_exc()
        with file_lock:
            with open(output_file, "a") as f:
                f.write(f"Error processing video: {video_path}. Error: {str(e)}\n\n")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Process videos with Mistral Vision API and Best-of-N strategy.")
    parser.add_argument(
        "--mode", choices=["discard", "continue"], default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos."
    )
    parser.add_argument(
        "--num_frames", type=int, default=8,
        help="Number of frames to sample from each video (default: 8)"
    )
    parser.add_argument(
        "--video_dir", type=str, default="data/exp/video_clips_6s_3",
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--fewshot", type=bool, default=False,
        help="Whether to use few-shot examples (default: False)"
    )
    parser.add_argument(
        "--fewshot_dir", type=str, default="data/exp/few_shot_examples_img",
        help="Directory containing few-shot example images"
    )
    parser.add_argument(
        "--output_file", type=str, default="data/exp/predictions/test_results_mistral.txt",
        help="File to save complete conversation logs"
    )
    parser.add_argument(
        "--json_output", type=str, default="data/exp/predictions/results_eval_mistral.json",
        help="JSON file to save evaluation metrics"
    )
    parser.add_argument(
        "--continue_file", type=str, default=None,
        help="JSON file from previous run to continue processing from"
    )
    parser.add_argument(
        "--bon_samples", type=int, default=5,
        help="Number of samples to generate for Best-of-N strategy (default: 5)"
    )
    args = parser.parse_args()

    # Record program start time
    start_time = time.time()

    # Generate timestamp filenames if not in continue mode
    if args.mode == "discard":
        args.output_file = generate_timestamp_filename(args.output_file)
        args.json_output = generate_timestamp_filename(args.json_output)
    
    # Print configuration
    print(f"Video directory: {args.video_dir}")
    print(f"Output file: {args.output_file}")
    print(f"JSON output file: {args.json_output}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Using few-shot learning: {args.fewshot}")
    print(f"Best-of-N samples: {args.bon_samples}")
    if args.fewshot:
        print(f"Few-shot examples directory: {args.fewshot_dir}")
    print(f"Mode: {args.mode}")

    # Check for continue mode
    processed_videos = set()
    continue_mode = False

    if args.mode == "continue" or args.continue_file:
        continue_mode = True
        continue_file = args.continue_file or args.json_output
        
        if os.path.exists(continue_file):
            processed_videos = get_processed_videos(continue_file)
            print(f"Continue mode: found {len(processed_videos)} already processed videos")
        else:
            print(f"Continue file {continue_file} not found, will process all videos")
    
    if args.mode == "discard":
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
        
        # Clear output files
        with open(args.output_file, 'w') as f:
            pass
        with open(args.json_output, 'w') as f:
            json.dump([], f)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)

    # Get list of videos to process
    video_files = [
        os.path.join(args.video_dir, filename)
        for filename in os.listdir(args.video_dir)
        if filename.endswith((".mp4", ".mov"))
    ]

    # Filter out already processed videos in 'continue' mode
    if continue_mode:
        video_files = [vf for vf in video_files if os.path.basename(vf) not in processed_videos]

    if not video_files:
        print("No videos to process.")
        return

    print(f"Found {len(processed_videos)} videos processed. Total videos to process: {len(video_files)}")

    # Tracking counts for summary
    successful_videos = 0
    failed_videos = 0
    skipped_videos = len(processed_videos)

    # Process videos with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_video, 
                video_path, 
                args.output_file, 
                args.json_output,
                args.fewshot_dir,
                args.fewshot,
                args.num_frames,
                args.bon_samples,
                args.continue_file if continue_mode else None
            ): video_path for video_path in video_files
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            video_path = futures[future]
            try:
                success = future.result()
                if success:
                    successful_videos += 1
                else:
                    failed_videos += 1
                
                print(f"Finished processing {os.path.basename(video_path)} ({i}/{len(video_files)})")
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                failed_videos += 1
    
    # Calculate total execution time
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    # Print execution summary
    print("\n" + "="*50)
    print("Execution Summary:")
    print(f"Total execution time: {format_time(total_execution_time)}")
    print(f"Videos: {successful_videos} successful, {failed_videos} failed, {skipped_videos} skipped")
    print("="*50)


if __name__ == "__main__":
    main()