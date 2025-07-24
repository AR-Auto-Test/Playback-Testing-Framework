"""
python 1_label_gpt_4o.py --mode discard (to start fresh)
python 1_label_gpt_4o.py --mode continue (to resume from where it left off)

# 明确启用 budget forcing
python 1_label_gpt_4o.py --mode discard --budget_forcing

# 禁用 budget forcing
python 1_label_gpt_4o.py --mode discard --budget_forcing=False
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

def generate_budget_forcing_context():
    """生成具有budget forcing特性的增强上下文"""
    base_context = generate_context()  # 获取基础上下文
    
    # 添加budget forcing的结构化分析指导
    budget_forcing_addition = """
    
    When providing your analysis, please follow these guidelines for thorough evaluation:
    
    1. First step: Carefully examine all video frames to identify any potential issues related to each metric.
    
    2. For each metric evaluation:
       - Take time to analyze multiple aspects of the AR effect
       - Consider both technical implementation quality and end-user experience
       - Look for subtle details that might indicate issues
       - Provide specific evidence from the frames to support your analysis
    
    3. Before concluding:
       - Double-check your reasoning for consistency
       - Consider alternative explanations for observed phenomena
       - Verify you haven't overlooked any important details
    
    4. Provide detailed, evidence-based reasoning in your JSON responses.
    """
    
    return base_context + budget_forcing_addition

def process_video(video_path, output_file, json_output_file, use_budget_forcing=True):
    """Processes a single video with GPT-4 Vision and logs both complete responses and JSON data."""
    try:
        base64Frames = read_video_frames(video_path, __INTERVAL)
        frame_images = list(map(lambda x: {"image": x, "resize": 768}, base64Frames))
        video_name = video_path.split('/')[-1]

        # 根据是否启用budget forcing选择上下文
        context_text = generate_budget_forcing_context() if use_budget_forcing else generate_context()

        # System message for context
        system_message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": context_text
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
                "content": "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video?"
            }
        ]
        
        params = {
            "model": "gpt-4o",
            "messages": initial_conversation,
            "temperature": 1,
            "max_tokens": 2048,
        }

        response = client.chat.completions.create(**params)
        answer1 = response.choices[0].message.content
        
        # Step 2: Get 3D object description
        initial_conversation.append({"role": "assistant", "content": answer1})
        initial_conversation.append({
            "role": "user",
            "content": "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it."
        })
        
        response = client.chat.completions.create(**params)
        answer2 = response.choices[0].message.content
        
        initial_conversation.append({"role": "assistant", "content": answer2})
        
        result_log = [
            f"Processing video: {video_path}\n\n",
            f"Question: What do you see in this video?\nAnswer: {answer1}\n{'-' * 20}\n",
            f"Question: What is the 3D object in the video?\nAnswer: {answer2}\n{'-' * 20}\n"
        ]
        
        # Step 3: Evaluate all AR aspects using budget forcing if enabled
        metrics = [
            "Object Placement",
            "Occlusion",
            "Object Movement",
            "Lighting",
            "Visual Artifacts and Rendering Issues",
            "Black Screen"
        ]
        
        json_results = []
        
        for metric in metrics:
            # 根据是否启用budget forcing创建不同的问题
            if use_budget_forcing:
                question = (
                    f"Is there any issue with the {metric}? Please analyze thoroughly:\n\n"
                    f"1. Carefully examine all frames for {metric.lower()} issues\n"
                    f"2. For each potential issue, analyze whether it's a real AR problem\n"
                    f"3. Consider both user experience and technical implementation\n"
                    f"4. Provide specific evidence from the frames in your analysis\n"
                    f"5. Cross-check your reasoning before concluding\n"
                    f"6. Provide your answer in JSON format with these fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'"
                )
            else:
                question = f"Is there any issue with the {metric}? Please clearly state your reason and give your answer in JSON format."
            
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
                        
                        # Add budget_forcing flag if enabled
                        if use_budget_forcing:
                            json_data["budget_forcing"] = True
                            
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
        
        # Write complete responses to log file
        with file_lock:
            with open(output_file, "a") as f:
                f.writelines(result_log)
                f.write("\n")
        
        # Determine appropriate output file based on budget_forcing
        if json_results:
            output_dir = os.path.dirname(json_output_file)
            
            # 使用原始json输出文件
            with file_lock:
                with open(json_output_file, "a") as f:
                    for result in json_results:
                        json.dump(result, f, indent=4)
                        f.write("\n")
            
            # 如果启用了budget forcing，也保存到专用文件
            if use_budget_forcing:
                bf_dir = os.path.join(output_dir, "budget_forcing_results")
                os.makedirs(bf_dir, exist_ok=True)
                bf_file = os.path.join(bf_dir, "label_gpt4o_bf.json")
                
                with file_lock:
                    with open(bf_file, "a") as f:
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
    video_dir = "../video_clips_6s_2"
    output_file = "log/result_gpt4o.txt"
    json_output_file = "json/label_gpt4o.json"

    parser = argparse.ArgumentParser(description="Process videos with GPT-4 Vision.")
    parser.add_argument(
        "--mode", choices=["discard", "continue"], default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos."
    )
    parser.add_argument(
        "--budget_forcing", action="store_true", default=True,
        help="Enable budget forcing to enhance reasoning"
    )
    args = parser.parse_args()

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
                        video_name = line.split("/")[-1].strip()
                        processed_videos.add(video_name)

        # Parse JSON file to track processed videos
        """if os.path.exists(json_output_file):
            with open(json_output_file, 'r') as f:
                for line in f:
                    try:
                        json_entry = json.loads(line)
                        processed_videos.add(json_entry.get("Video_name", ""))
                    except json.JSONDecodeError:
                        continue"""

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

    print(f"Found {len(processed_videos)} video processed. Total videos to process: {len(video_files)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_video, video_path, output_file, json_output_file, args.budget_forcing): video_path for video_path in video_files}
        
        for i, future in enumerate(as_completed(futures), start=1):
            video_path = futures[future]
            try:
                future.result()
                print(f"Finished processing {video_path} ({i}/{len(video_files)})")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    main()
