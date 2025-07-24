"""
python 3_label_gemini2_pro.py --mode discard (to start fresh)
python 3_label_gemini2_pro.py --mode continue (to resume from where it left off)

# 明确启用 budget forcing
python 3_label_gemini2_pro.py --mode discard --budget_forcing

# 禁用 budget forcing
python 3_label_gemini2_pro.py --mode discard --budget_forcing=False
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
        choices=["discard", "continue"], 
        default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos."
    )
    parser.add_argument(
        "--budget_forcing",
        action="store_true",
        default=True,
        help="Whether to use budget forcing for enhanced reasoning."
    )
    return parser.parse_args()

def generate_budget_forcing_context():
    """生成带有budget forcing增强的上下文提示"""
    base_context = generate_context()  # 获取基础上下文
    
    # 添加budget forcing特定的指导
    budget_forcing_addition = """
    
    IMPORTANT EVALUATION GUIDELINES:
    
    When analyzing AR effects, please follow these steps for thorough evaluation:
    
    1. Take your time to carefully observe all video frames first, paying close attention to all details.
    
    2. When evaluating each metric:
       - First identify all potential issues related to that metric
       - For each potential issue, examine it from multiple perspectives
       - Consider both technical implementation and user experience impacts
       - Provide specific evidence from the video frames to support your analysis
    
    3. Before making a final judgment:
       - Double-check your reasoning for logical consistency
       - Consider alternative explanations for observed phenomena
       - Verify you haven't overlooked any important details
    
    4. Provide detailed, evidence-based reasoning in your JSON responses.
    """
    
    return base_context + budget_forcing_addition

def load_processed_videos(output_file, json_output_file):
    """Load list of already processed videos from both output files."""
    processed_videos = set()
    
    # Check log file
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.startswith("Processing video:"):
                    video_path = line.split("Processing video:")[1].strip()
                    processed_videos.add(os.path.basename(video_path))
    
    # Check JSON file
    if os.path.exists(json_output_file):
        with open(json_output_file, 'r') as f:
            for line in f:
                try:
                    json_data = json.loads(line.strip())
                    if 'Video_name' in json_data:
                        processed_videos.add(json_data['Video_name'])
                except json.JSONDecodeError:
                    continue
    
    return processed_videos

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
    
def generate_context():
    return """You are a specialized AI assistant trained to review AR applications based on screen recordings. The video provided is a screen recording from a mobile phone that demonstrates an AR (Augmented Reality) application. We specifically designed and selected certain environment and use cases to test the AR application to identify any potential performance issues with AR functionality. In all the screen recording videos, the movement of AR objects is simulated programmatically as interactions on the phone screen. Currently, the simulated interactions include the following two types: 1.Tapping the screen: The AR object will immediately appear at the corresponding position. 2. Swiping the screen: The AR object will move along the swipe path to the corresponding positions. 
    
    Your task is to evaluate the realism of AR effect and the performance of AR apps based on following metrics: 1. Object Placement. 2. Object Movement. 3. Occlusion. 4. Lighting. 5. Visual Artifacts and Rendering Issues. 6. Black Screen 
    
    And you could ignore the following aspects in evaluation: 
    1. The size of AR object. Because some apps may privide the resize feature so that the user could adjust. 
    2. The stylish or cartoonish AR object. Because some apps may provide stylish and cartoonish models according to the theme of app. The The stylish or cartoonish does not account for realism issue. 
    3. The UI elements. The effect and visibility of UI elements like grid lines or measures are part of design by developers. Therefore, the interaction between AR object and UI element is out of the scope of evaluation.
    
    For evaluation questions, I will ask you to response in JSON format. Please ensure your response is a valid JSON object with this format:
    {{
        "Video_name": string,   
        "Metrics": string, // the metric to evaluate, e.g. Object Placement
        "Issue": boolean, // true if there are issues found 
        "Reason":  string //The reason why you think there are issues or no issues. Provide explanation.
    }}
    """

def process_video(video_path, model, output_file, json_output_file, use_budget_forcing=True):
    """Processes a single video with Gemini and logs both complete responses and JSON data."""
    try:
        video_name = os.path.basename(video_path)
        
        # Wrap the upload in safe_gemini_request
        uploaded_file = safe_gemini_request(
            upload_to_gemini, 
            video_path, 
            mime_type="video/mp4"
        )
        
        wait_for_files_active([uploaded_file])

        # 选择使用基础上下文或增强的budget forcing上下文
        context = generate_budget_forcing_context() if use_budget_forcing else generate_context()

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
            f"The video name is {video_name}.What do you see in this video? Can you see the AR effect in the video?",
            "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description."
        ]

        metrics = [
            "Object Placement",
            "Occlusion", 
            "Object Movement",
            "Lighting",
            "Visual Artifacts and Rendering Issues",
            "Black Screen"
        ]

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

        # Process metrics with JSON responses and budget forcing if enabled
        for metric in metrics:
            # 根据是否启用budget forcing创建不同的问题
            if use_budget_forcing:
                question = (
                    f"Is there any issue with the {metric}? Please provide a thorough analysis:\n\n"
                    f"1. Carefully observe all frames for potential {metric.lower()} issues\n"
                    f"2. For each potential issue, evaluate whether it's a real AR problem\n"
                    f"3. Consider this from multiple perspectives (user experience, technical implementation)\n"
                    f"4. Cross-check your reasoning before finalizing your answer\n"
                    f"5. Provide your answer in JSON format with fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'"
                )
            else:
                question = f"Is there any issue with the {metric}? Please provide your answer in JSON format with the following fields: 'Metrics', 'Issue' (boolean), and 'Reason'."

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    response = safe_gemini_request(
                        lambda: model.start_chat(history=chat_history).send_message(question)
                    )

                    answer = response.parts[0].text if response.parts else "No response"
                    result_log.append(f"Question: {question}\nAnswer: {answer}\n{'-' * 20}\n")

                    # Try to extract and validate JSON from response
                    import re
                    json_match = re.search(r"\{[\s\S]*\}", answer)
                    if json_match:
                        json_str = json_match.group(0)
                        json_data = json.loads(json_str)

                        # Add or update Video_name in the JSON data
                        json_data["Video_name"] = video_name
                        # Add whether budget forcing was used
                        if use_budget_forcing:
                            json_data["budget_forcing"] = True

                        # Validate JSON structure
                        if all(key in json_data for key in ["Metrics", "Issue", "Reason"]):
                            json_results.append(json_data)
                            break

                    retry_count += 1
                    time.sleep(2)
                except Exception as e:
                    print(f"Error processing metric {metric}: {str(e)}")
                    retry_count += 1
                    time.sleep(2)

        # Write results with file locks
        with file_lock:
            with open(output_file, "a") as f:
                f.writelines(result_log)
                f.write("\n")

        if json_results:
            # 确定输出文件路径
            actual_json_output = json_output_file
            if use_budget_forcing:
                # 创建特定于budget forcing的输出文件
                bf_dir = os.path.dirname(json_output_file)
                bf_file = os.path.join(bf_dir, "label_gemini2_pro_bf.json")
                actual_json_output = bf_file
                
                # 确保目录存在
                os.makedirs(os.path.dirname(actual_json_output), exist_ok=True)
            
            with file_lock:
                with open(actual_json_output, "a") as f:
                    for result in json_results:
                        json.dump(result, f, indent=4)
                        f.write("\n")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        with file_lock:
            with open(output_file, "a") as f:
                f.write(f"Error processing video: {video_path}. Error: {e}\n\n")

            
def main():
    args = parse_args()
    
    video_dir = "../video_clips_6s_2_for_gemini"
    output_file = "log/result_gemini2_pro.txt"
    json_output_file = "json/label_gemini2_pro.json"

    # 如果使用budget forcing，创建专用目录
    if args.budget_forcing:
        bf_dir = "budget_forcing_results"
        os.makedirs(bf_dir, exist_ok=True)
    
    # Handle different modes
    if args.mode == "discard":
        # Clear output files
        for file_path in [output_file, json_output_file]:
            with open(file_path, 'w') as f:
                pass
        processed_videos = set()
    else:  # continue mode
        processed_videos = load_processed_videos(output_file, json_output_file)
        print(f"Found {len(processed_videos)} previously processed videos")
    
    # Setup Gemini model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-pro-exp-02-05",
        generation_config=generation_config,
    )

    # Process videos
    video_files = [f for f in os.listdir(video_dir) 
                  if f.endswith((".mp4", ".mov"))]
    
    if args.mode == "continue":
        video_files = [f for f in video_files 
                      if f not in processed_videos]
    
    total_videos = len(video_files)
    print(f"Found {total_videos} videos to process")
    print(f"Budget forcing enabled: {args.budget_forcing}")
    
    for i, filename in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, filename)
        print(f"\nProcessing video {i}/{total_videos}: {filename}")
        process_video(video_path, model, output_file, json_output_file, use_budget_forcing=args.budget_forcing)

if __name__ == "__main__":
    main()