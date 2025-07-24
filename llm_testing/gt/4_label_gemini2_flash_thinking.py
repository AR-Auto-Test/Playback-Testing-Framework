"""
python 4_label_gemini2_flash_thinking.py --mode discard (to start fresh)
python 4_label_gemini2_flash_thinking.py --mode continue (to resume from where it left off)
python 4_label_gemini2_flash_thinking.py --mode update --update_metric "Lighting" (to only update a specific metric)
"""

import os
import time
import json
import argparse
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import Content
from dotenv import load_dotenv
from threading import Lock
import random
from datetime import datetime, timedelta

# Import the new prompts module
from ar_prompts_labels import generate_system_prompt, generate_ar_metrics_description, get_metrics, METRICS

load_dotenv()  # Load environment variables from .env file

# Constants for rate limiting
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10  # seconds
MAX_RETRY_DELAY = 300    # 5 minutes
RATE_LIMIT_WINDOW = 60   # 1 minute
MAX_REQUESTS_PER_MINUTE = 30  # Adjust based on your quota

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Add file lock for thread-safe file operations
file_lock = Lock()

class RateLimiter:
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = Lock()

    def wait_if_needed(self):
        current_time = datetime.now()
        with self.lock:
            # Remove requests older than our window
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < timedelta(seconds=RATE_LIMIT_WINDOW)]
            
            if len(self.requests) >= self.requests_per_minute:
                # Calculate sleep time needed
                oldest_request = min(self.requests)
                sleep_time = (oldest_request + timedelta(seconds=RATE_LIMIT_WINDOW) - current_time).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Add current request
            self.requests.append(current_time)

rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

def exponential_backoff(retry_count):
    """Calculate exponential backoff time with jitter."""
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** retry_count))
    jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
    return delay + jitter

def handle_rate_limit_error(retry_count):
    """Handle rate limit error with exponential backoff."""
    if retry_count >= MAX_RETRIES:
        raise Exception("Max retries exceeded")
    
    delay = exponential_backoff(retry_count)
    print(f"Rate limit exceeded. Waiting {delay:.2f} seconds before retry {retry_count + 1}/{MAX_RETRIES}")
    time.sleep(delay)

def safe_gemini_request(func, *args, **kwargs):
    """Execute a Gemini API request with rate limiting and error handling."""
    retry_count = 0
    while True:
        try:
            rate_limiter.wait_if_needed()
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                retry_count += 1
                handle_rate_limit_error(retry_count)
            else:
                raise

def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with Gemini Vision API.")
    parser.add_argument(
        "--mode", 
        choices=["discard", "continue", "update"], 
        default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos, 'update' to update specific metrics."
    )
    parser.add_argument(
        "--update_metric",
        type=str,
        help="Specify which metric to update (only used with --mode update)."
    )
    return parser.parse_args()

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

def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()

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
    
    # Read all JSON objects from the file
    with open(json_output_file, 'r') as f:
        content = f.read()
    
    # Parse JSON objects
    json_objects = []
    filtered_objects = []
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
    
    # Filter out objects for the specified metric
    for json_str in json_objects:
        try:
            json_data = json.loads(json_str)
            if 'Metrics' in json_data and json_data['Metrics'] == metric_to_update:
                # Skip this object (it's for the metric we want to update)
                continue
            filtered_objects.append(json_str)
        except json.JSONDecodeError:
            # Keep invalid JSON objects (unlikely, but just in case)
            filtered_objects.append(json_str)
    
    # Write filtered objects back to the file
    with open(json_output_file, 'w') as f:
        for json_str in filtered_objects:
            f.write(json_str)
            f.write('\n')
    
    print(f"Filtered out entries for metric '{metric_to_update}' from the JSON file")

def process_video(video_path, model, output_file, json_output_file, processed_metrics=None, update_metric=None):
    """Processes a single video with Gemini and logs both complete responses and JSON data."""
    try:
        video_name = os.path.basename(video_path)
        
        # Get the set of metrics already processed for this video
        metrics_to_skip = set()
        if processed_metrics and video_name in processed_metrics:
            metrics_to_skip = processed_metrics[video_name]
            if metrics_to_skip:
                print(f"Skipping previously processed metrics for {video_name}: {', '.join(metrics_to_skip)}")
        
        # If we're updating a specific metric, ensure it's not in the skip list
        if update_metric:
            if update_metric in metrics_to_skip:
                metrics_to_skip.remove(update_metric)
                print(f"Will re-evaluate metric '{update_metric}' for {video_name}")
        
        # Wrap the upload in safe_gemini_request
        uploaded_file = safe_gemini_request(
            upload_to_gemini, 
            video_path, 
            mime_type="video/mp4"
        )
        
        wait_for_files_active([uploaded_file])

        # Get context from ar_prompts_labels
        context = generate_system_prompt()

        chat_history = [
            {
                "role": "user",
                "parts": [uploaded_file],
            },
            {
                "role": "user",
                "parts": [
                    context
                ],
            },
        ]

        initial_questions = [
            f"The video name is {video_name}. What do you see in this video? Can you see the AR effect in the video? Can you describe the setting (indoor/outdoor), the primary surfaces or objects in view, and any notable lighting conditions?",
            "What is the 3D object in the video? If you can see the 3D object, describe it in detail including its type, color, and any notable features. If you are not clear what exactly the object is, try to describe it as specifically as possible."
        ]

        # Use metrics from ar_prompts_labels
        metrics = METRICS

        result_log = [f"Processing video: {video_path}\n\n"]
        json_results = []

        # Process initial questions with rate limiting
        for question in initial_questions:
            response = safe_gemini_request(
                lambda: model.start_chat(history=chat_history).send_message(question)
            )
            
            answer = response.parts[0].text if response.parts else "No response"
            result_log.append(f"Question: {question}\nAnswer: {answer}\n{'-' * 20}\n")
            
            chat_history.append({"role": "user", "parts": [question]})
            chat_history.append({"role": "model", "parts": [answer]})

        # Process metrics with JSON responses
        for i, metric in enumerate(metrics):
            # If update_metric is specified, only process that metric
            if update_metric and metric != update_metric:
                continue
                
            # Skip already processed metrics (except the one we want to update)
            if metric in metrics_to_skip:
                print(f"  Skipping already processed metric: {metric}")
                continue
                
            print(f"  Processing metric: {metric}")
            
            # Get detailed description for the metric
            metric_description = generate_ar_metrics_description(i)
            question = f"Is there any issue with the {metric}? \n{metric_description}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'."

            max_retries = 3
            retry_count = 0
            last_error = None
            last_response = None

            while retry_count < max_retries:
                try:
                    response = safe_gemini_request(
                        lambda: model.start_chat(history=chat_history).send_message(question)
                    )

                    answer = response.parts[0].text if response.parts else "No response"
                    last_response = answer  # Store the last response
                    result_log.append(f"Question: {question}\nAnswer: {answer}\n{'-' * 20}\n")

                    # Try to extract and validate JSON from response
                    import re
                    json_match = re.search(r"\{[^{}]*\}", answer)
                    if json_match:
                        json_str = json_match.group(0)
                        json_data = json.loads(json_str)

                        # Add or update Video_name in the JSON data
                        json_data["Video_name"] = video_name
                        
                        # Add or update Metrics in the JSON data if not present
                        if "Metrics" not in json_data:
                            json_data["Metrics"] = metric

                        # Validate JSON structure
                        if all(key in json_data for key in ["Metrics", "Issue", "Reason"]):
                            json_results.append(json_data)
                            break

                    retry_count += 1
                    time.sleep(2)
                except Exception as e:
                    print(f"Error processing metric {metric}: {str(e)}")
                    last_error = e
                    retry_count += 1
                    time.sleep(2)
            
            # Log detailed error if all retries failed
            if retry_count == max_retries:
                error_message = f"Failed to get valid JSON response for {metric} after {max_retries} attempts\n"
                if last_response:
                    error_message += f"Last response:\n{last_response}\n"
                if last_error:
                    import traceback
                    error_traceback = traceback.format_exc()
                    error_message += f"Last error traceback:\n{error_traceback}\n"
                
                result_log.append(f"ERROR for {metric}:\n{error_message}\n{'-' * 20}\n")
                print(error_message)

        # Write results with file locks
        with file_lock:
            with open(output_file, "a") as f:
                f.writelines(result_log)
                f.write("\n")

        if json_results:
            with file_lock:
                with open(json_output_file, "a") as f:
                    for result in json_results:
                        json.dump(result, f, indent=4)
                        f.write("\n")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        with file_lock:
            with open(output_file, "a") as f:
                f.write(f"Error processing video: {video_path}. Error: {e}\n")
                f.write(f"Error traceback:\n{error_traceback}\n\n")

            
def main():
    args = parse_args()
    
    video_dir = "../video_clips_6s_2_for_gemini"
    output_file = "log/result_gemini2_flash_thinking.txt"
    json_output_file = "json/label_gemini2_flash_thinking.json"

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

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
    
    # Handle different modes
    if args.mode == "discard":
        # Clear output files
        with open(json_output_file, 'w') as f:
            pass
        with open(output_file, 'w') as f:
            pass
        processed_metrics = {}
    elif args.mode == "update":
        # Create backup of existing results
        backup_json_file(json_output_file)
        # Load existing metrics first
        processed_metrics = load_processed_videos(json_output_file)
        # Filter out entries for the metric we want to update
        filter_json_file(json_output_file, update_metric)
        # Create a header in the log file to indicate this is an update run
        with open(output_file, 'a') as f:
            f.write(f"\n{'-' * 80}\n")
            f.write(f"UPDATE RUN FOR METRIC: {update_metric} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'-' * 80}\n\n")
    else:  # continue mode
        processed_metrics = load_processed_videos(json_output_file)
    
    # Setup Gemini model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro-preview-03-25",
        generation_config=generation_config,
    )

    # Get all video files
    all_video_files = [f for f in os.listdir(video_dir) 
                     if f.endswith((".mp4", ".mov"))]
    
    # Create a list of videos that need processing
    videos_to_process = []
    for filename in all_video_files:
        video_name = filename
        
        # For update mode, include all videos regardless of whether they have all metrics processed
        if args.mode == "update":
            videos_to_process.append(filename)
            continue
            
        # For continue mode, skip videos with all 6 metrics processed
        if video_name in processed_metrics and len(processed_metrics[video_name]) == 6:
            continue
        
        videos_to_process.append(filename)
    
    if args.mode == "update":
        print(f"Update mode: Will process {update_metric} for all {len(videos_to_process)} videos")
    else:
        print(f"Found {len(videos_to_process)} videos that need additional processing out of {len(all_video_files)} total videos")
    
    for i, filename in enumerate(videos_to_process, 1):
        video_path = os.path.join(video_dir, filename)
        if args.mode == "update":
            print(f"\nProcessing video {i}/{len(videos_to_process)}: {filename} - updating metric '{update_metric}'")
        else:
            print(f"\nProcessing video {i}/{len(videos_to_process)}: {filename}")
        
        process_video(video_path, model, output_file, json_output_file, 
                      processed_metrics=processed_metrics,
                      update_metric=update_metric)

if __name__ == "__main__":
    main()