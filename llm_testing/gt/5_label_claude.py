"""
python 5_label_claude.py --mode discard (to start fresh)
python 5_label_claude.py --mode continue (to resume from where it left off)
"""

import cv2
import base64
import time
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import traceback
import argparse
import json
import re

load_dotenv()  # Load environment variables from .env file

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-3-5-sonnet-20241022"
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

def generate_context():
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

def process_video(video_path, output_file, json_output_file):
    """Processes a single video with Claude and logs both complete responses and JSON data."""
    try:
        base64Frames = read_video_frames(video_path, __INTERVAL)
        video_name = os.path.basename(video_path)

        # Initial conversation to establish context
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The video name is {video_name}. These are frames from a video that I want to upload. Please check them and I will ask questions later."
                    }
                ]
            }
        ]
        
        # Add image content
        for frame in base64Frames:
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame
                }
            })
            
        

        # Initial description question
        messages.append({
            "role": "user",
            "content": "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video?"
        })
        
        params = {
            "model": MODEL,
            "system": generate_context(),
            "messages": messages,
            "temperature": 1,
            "max_tokens": 2048,
        }

        response = client.messages.create(**params)
        answer1 = response.content[0].text

        # 3D object description question
        messages.append({"role": "assistant", "content": answer1})
        messages.append({
            "role": "user",
            "content": "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it."
        })

        response = client.messages.create(**params)
        answer2 = response.content[0].text

        messages.append({"role": "assistant", "content": answer2})

        result_log = [
            f"Processing video: {video_path}\n\n",
            f"Question: What do you see in this video?\nAnswer: {answer1}\n{'-' * 20}\n",
            f"Question: What is the 3D object in the video?\nAnswer: {answer2}\n{'-' * 20}\n"
        ]

        # Evaluate all AR aspects
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
            question = f"Is there any issue with the {metric}? Please clearly state your reason and give your answer in JSON format."
            
            single_message = messages + [
                {"role": "user", "content": question}
            ]
            params["messages"] = single_message
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                response = client.messages.create(**params)
                answer = response.content[0].text
                
                # Save complete response to log
                result_log.append(f"Question: {question}\nAnswer: {answer}\n{'-' * 20}\n")
                
                # Try to extract JSON from response
                try:
                    json_match = re.search(r"\{[^{}]*\}", answer)
                    if json_match:
                        json_str = json_match.group(0)
                        json_data = json.loads(json_str)
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

            messages.append({"role": "assistant", "content": answer})

        # Write complete responses to log file
        with open(output_file, "a") as f:
            f.writelines(result_log)
            f.write("\n")
        
        # Write JSON results to JSON file
        if json_results:
            with open(json_output_file, "a") as f:
                for result in json_results:
                    json.dump(result, f, indent=4)
                    f.write("\n")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        traceback.print_exc()
        with open(output_file, "a") as f:
            f.write(f"Error processing video: {video_path}. Error: {e}\n\n")

def main():
    video_dir = "../video_clips_6s_2"
    output_file = "log/result_claude.txt"
    json_output_file = "json/label_claude.json"

    parser = argparse.ArgumentParser(description="Process videos with Claude.")
    parser.add_argument(
        "--mode", choices=["discard", "continue"], default="discard",
        help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos."
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
                        video_name = line.split("Processing video:")[1].strip()
                        processed_videos.add(video_name)

        # Parse JSON file to track processed videos
        if os.path.exists(json_output_file):
            with open(json_output_file, 'r') as f:
                for line in f:
                    try:
                        json_entry = json.loads(line)
                        processed_videos.add(json_entry.get("Video_name", ""))
                    except json.JSONDecodeError:
                        continue

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

    print(f"Total videos to process: {len(video_files)}")

    # Process videos sequentially
    for i, video_path in enumerate(video_files, 1):
        try:
            print(f"Processing {video_path} ({i}/{len(video_files)})")
            process_video(video_path, output_file, json_output_file)
            print(f"Finished processing {video_path}")
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    main()