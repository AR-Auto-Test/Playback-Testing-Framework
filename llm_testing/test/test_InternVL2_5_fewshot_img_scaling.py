"""
# 标准执行（从头开始，不继续之前的任务）
python test_InternVL2_5_fewshot_img_scaling.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "OpenGVLab/InternVL2_5-8B-MPO" \
  --output_file responses/output.txt \
  --json_output responses/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 

# 继续执行（继续处理之前未完成的视频）
python test_InternVL2_5_fewshot_img_scaling.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "OpenGVLab/InternVL2_5-8B-MPO" \
  --output_file results/output.txt \
  --json_output results/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --continue_file results/evaluation_20240508_143022.json
"""

import os
import torch 
import av
import numpy as np
import argparse
import math
import json
import re
import datetime
import traceback
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406) 
IMAGENET_STD = (0.229, 0.224, 0.225)


class FewShotExample:
    def __init__(self, image_path, ground_truth):
        self.image_path = image_path
        self.ground_truth = ground_truth
        self.processed_image = None
        self.num_patches = None

    def load_image(self, transform, max_num=1):
        image = Image.open(self.image_path).convert('RGB')
        processed = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in processed]
        self.processed_image = torch.stack(pixel_values)
        self.num_patches = [self.processed_image.shape[0]]
        self.processed_image = self.processed_image.to(torch.bfloat16).cuda()
        return self.processed_image, self.num_patches


# 在 parse_args 函数中修改参数定义:
def parse_args():
    parser = argparse.ArgumentParser(description="Test InternVL2_5 model with image-based few-shot learning")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing test video files")
    parser.add_argument("--few_shot_dir", type=str, required=True, help="Directory containing few-shot example images")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="JSON file containing ground truth for few-shot examples")
    parser.add_argument("--output_file", type=str, default="output.txt", help="File to save complete conversation logs")
    parser.add_argument("--json_output", type=str, default="evaluation.json", help="JSON file to save evaluation metrics")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL2_5-8B", help="Model name or path")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to sample from test videos")
    # 替换 mode 参数
    parser.add_argument("--continue_file", type=str, default=None, 
                       help="JSON file from previous run to continue processing from. If provided and exists, will skip processed videos.")
    parser.add_argument("--bon_samples", type=int, default=16, 
                       help="Number of samples to generate for Best-of-N strategy")
    return parser.parse_args()

def load_few_shot_examples(few_shot_dir, ground_truth_file, transform):
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    examples = []
    image_files = [f for f in os.listdir(few_shot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Found {len(image_files)} few-shot example images in {few_shot_dir}")
    
    for image_file in image_files:
        image_path = os.path.join(few_shot_dir, image_file)
        if image_file in ground_truth_data:
            example = FewShotExample(
                image_path=image_path,
                ground_truth=ground_truth_data[image_file]
            )
            example.load_image(transform)
            examples.append(example)
        else:
            print(f"Warning: No ground truth found for {image_file}")
    
    return examples

def generate_few_shot_prompt(examples):
    prompt = "Here are some example AR screenshots with their quality assessment:\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Screenshot: <image>\n"
        prompt += f"Description:\n{example.ground_truth}\n\n"
    
    prompt += "Now, please understand the context of your task, and fully understand examples provided. Next, I will ask questions and please evaluate the AR effect in the Test Video.\n"
    return prompt

def generate_ar_metrics_description(metric):
    """Generate detailed descriptions for AR evaluation metrics."""
    metrics_des = [
        # 1.Object Placement
        """
        Evaluate whether AR objects are positioned correctly in the real environment, including:
        - Objects properly adhering to surfaces (tables, floors, walls)
        - No unintentional floating or hovering above surfaces
        - Objects respecting real-world physical space rules (not passing through solid surfaces)
        - Contact points between objects and surfaces appearing natural and logical
        - Objects maintaining consistent position when viewed from different angles or distances
        """,
        # 2.Object Movement
        """
        Evaluate whether AR objects move naturally and smoothly, including:
        - Movement appearing fluid without stuttering, jumping, or discontinuity
        - Objects maintaining shape and posture during movement without distortion
        - Objects maintaining correct spatial relationships with the scene while moving
        - Objects stabilizing after movement stops without drifting or oscillating
        """,
        # 3.Occlusion
        """Evaluate whether AR objects correctly interact with other objects regarding visibility, including:
        - Virtual objects being properly hidden when they should be occluded by real objects
        - Virtual objects correctly occluding real objects when positioned in front
        - Occlusion edges appearing precise without obvious clipping errors or gaps
        - People or moving objects in the environment correctly occluding AR objects
        - Objects at different distances showing correct front/back occlusion relationships
        - Occlusion effects remaining stable without suddenly appearing or disappearing with perspective changes
        - Occlusion relationships updating dynamically when AR objects move and interact with the environment
        """,
        # 4.Lighting
        """Evaluate whether AR objects' lighting effects match the surrounding environment, including:
        - Object brightness/darkness matching ambient lighting conditions
        - Objects casting shadows if there are environmental light sources
        - Shadow softness/hardness corresponding to environmental lighting (e.g., direct sunlight produces hard shadows, diffuse light produces soft shadows)
        - Objects displaying appropriate highlights and reflections consistent with material properties
        - Lighting effects on objects updating appropriately when environmental lighting changes
        """,
        # 5.Visual Artifacts and Rendering Issues
        """Evaluate any visual defects or abnormalities in AR rendering, including:
        - Object edges free from jaggedness, blurring, or flickering
        - Textures loading correctly without display errors or distortion
        - Objects free from unnatural shine or oversaturated colors
        - Objects free from flickering, jittering, or other unstable visual effects
        - Object materials displaying correctly (e.g., metals having metallic sheen, glass having transparency)
        - Objects free from unnatural geometric deformation or stretching
        - No sudden visual glitches or image corruption
        """,
        # 6.Black Screen
        """Evaluate whether the AR application suffers from black screens or other serious display issues, including:
        - Screen completely or partially black
        - AR objects suddenly disappearing under certain circumstances
        - Screen displaying large areas of white or other abnormal colors
        - Screen freezing or failing to update in large areas
        - AR application crashing or displaying abnormally after certain interactions
        - Screen displaying unintended color blocks or patterns
        - Camera feed failing to display properly or exhibiting severe delay
        - Obvious synchronization issues between AR layer and camera layer
        """
    ]
    return metrics_des[metric]

def generate_conversation(few_shot_prompt):
    """Generate conversation with few-shot context."""
    base_system_message = f"""You are a specialized AI assistant trained to review AR applications based on screen recordings. The video provided is a screen recording from a mobile phone that demonstrates an AR (Augmented Reality) application. We specifically designed and selected certain environment and use cases to test the AR application to identify any potential performance issues with AR functionality. In all the screen recording videos, the movement of AR objects is simulated programmatically as interactions on the phone screen. Currently, the simulated interactions include the following two types: 1.Tapping the screen: The AR object will immediately appear at the corresponding position. 2. Swiping the screen: The AR object will move along the swipe path to the corresponding positions. 
    
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
        "Reason":  string //The reason why you think there are issues or no issues. Provide explanation as clear as possible.
    }}
    """
    
    full_system_message = base_system_message + "\n\n" + few_shot_prompt
    
    conversation = [
        full_system_message,
        "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video? You don't need to do the evaluation here now.",
        "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it.",
        f"Is there any issue with the Object Placement? \n{generate_ar_metrics_description(0)}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        f"Is there any issue with the Object Movement? \n{generate_ar_metrics_description(1)}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.", 
        f"Is there any issue with the Occlusion? \n{generate_ar_metrics_description(2)}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'. ",
        f"Is there any issue with the Lighting? \n{generate_ar_metrics_description(3)}\n Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        f"Is there any issue with Visual Artifacts and Rendering Issues? \n{generate_ar_metrics_description(4)}\n  Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        f"Is there any issue of Black Screen? \n{generate_ar_metrics_description(5)}\n  Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        "Except Object Placement, Object Movement, Occlusion, Lighting, Artifacts and Rendering and Black Screen, have you found any other issues about Augmented Reality in this video?"
    ]
    
    return conversation

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size=448):
    transform = Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

# video multi-round conversation (视频多轮对话)
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPUs.")
    else:
        device = torch.device("cpu")
        print("No GPUs ound, using CPU.")
    return device

def extract_json_content(text):
    """Extract JSON content from model response with improved pattern matching."""
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

def get_diverse_params(sample_index, total_samples):
    # 根据样本索引生成多样化的参数
    temperature = 0.5 + (0.5 * sample_index / total_samples)  # 从0.5到1.0变化
    top_p = 0.85 + (0.1 * sample_index / total_samples)  # 从0.85到0.95变化
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 40,
        "max_new_tokens": 2048
    }


# 修改 test_multi_turn_with_few_shot 函数，实现Best-of-N的TTS策略
def test_multi_turn_with_few_shot(model, tokenizer, test_frames, test_num_patches, few_shot_examples, conversation, device, video_file, bon_samples=16):
    model.eval()
    responses = []
    evaluation_results = {}
    generation_config = dict(max_new_tokens=1024, do_sample=True, top_k=40, top_p=0.95, temperature=0.7)  # 添加采样参数
    
    # 结合示例帧和测试视频帧
    all_frames = []
    all_num_patches = []
    for example in few_shot_examples:
        all_frames.append(example.processed_image)
        all_num_patches.extend(example.num_patches)
    all_frames.append(test_frames)
    all_num_patches.extend(test_num_patches)
    combined_frames = torch.cat(all_frames)
    
    # 初始验证问题 (context和前两个问题)
    history = None
    for turn in range(3):  # Context和前两个验证问题
        question = conversation[turn]
        if turn == 0:
            frame_prefix = []
            frame_prefix.extend([f'Test Video Frame{i+1}: <image>\n' 
                               for i in range(len(test_num_patches))])
            question = ''.join(frame_prefix) + question
        
        response, history = model.chat(
            tokenizer,
            combined_frames,
            question,
            generation_config,
            num_patches_list=all_num_patches,
            history=history,
            return_history=True
        )
        responses.append(response)
    
    # 保存初始历史记录
    initial_history = history  
    
    # 六个评估问题 (使用Best-of-N策略)
    evaluation_responses = []
    metrics = ["Object Placement", "Object Movement", "Occlusion", 
               "Lighting", "Visual Artifacts", "Black Screen"]
    
    for i in range(6):  # 六个评估问题
        history = initial_history  # 重置历史记录到验证问题后
        question = conversation[i + 3]  # 评估问题的偏移量为3
        
        # 使用Best-of-N策略生成多个回答
        bon_responses = []
        bon_json_data = []
        
        # 生成N个样本
        for n in range(bon_samples):
            params = get_diverse_params(n, bon_samples)
            response, _ = model.chat(
                tokenizer,
                combined_frames,
                question,
                params,  # 使用多样化的参数
                num_patches_list=all_num_patches,
                history=history,
                return_history=True
            )
            bon_responses.append(response)
            
            # 尝试解析JSON响应
            try:
                json_content = extract_json_content(response)
                if json_content:
                    json_data = json.loads(json_content)
                    json_data['Video_name'] = video_file
                    bon_json_data.append(json_data)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for sample {n} of metric {metrics[i]}")
        
        # 选择最佳回答 (使用多数投票)
        best_response, best_json = select_best_response(bon_responses, bon_json_data)
        evaluation_responses.append(best_response)
        
        if best_json:
            evaluation_results[metrics[i]] = best_json
        else:
            evaluation_results[metrics[i]] = {
                "error": "No valid JSON found in any sample",
                "raw_response": best_response
            }
    
    # 最后的扩展问题 (不使用BoN)
    history = initial_history
    for i, response in enumerate(evaluation_responses):
        history.append({"role": "user", "content": conversation[i + 3]})
        history.append({"role": "assistant", "content": response})
    
    final_question = conversation[-1]
    final_response, _ = model.chat(
        tokenizer,
        combined_frames,
        final_question,
        generation_config,
        num_patches_list=all_num_patches,
        history=history,
        return_history=True
    )
    
    responses.extend(evaluation_responses)
    responses.append(final_response)
    
    return responses, evaluation_results

# 辅助函数：从多个样本中选择最佳回答
def select_best_response(responses, json_data):
    # 如果没有成功解析的JSON，返回第一个响应
    if not json_data:
        return responses[0], None
    
    # 计算每个"Issue"值的频率
    issue_counts = {"true": 0, "false": 0}
    valid_responses = []
    valid_json = []
    
    for i, json_obj in enumerate(json_data):
        if "Issue" in json_obj:
            issue_value = str(json_obj["Issue"]).lower()
            issue_counts[issue_value] = issue_counts.get(issue_value, 0) + 1
            valid_responses.append(responses[i])
            valid_json.append(json_obj)
    
    # 如果没有有效的响应，返回第一个
    if not valid_responses:
        return responses[0], None
    
    # 确定多数投票的结果
    majority_issue = "true" if issue_counts.get("true", 0) > issue_counts.get("false", 0) else "false"
    
    # 找到与多数投票结果匹配的第一个响应
    for i, json_obj in enumerate(valid_json):
        if str(json_obj["Issue"]).lower() == majority_issue:
            return valid_responses[i], json_obj
    
    # 默认返回第一个有效响应
    return valid_responses[0], valid_json[0]

def main():
    args = parse_args()
    device = setup_device()
    
    scale = args.model_name.split("-")[1]
    device_map = split_model(f'InternVL2_5-{scale}')
    transform = build_transform(input_size=448)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 修改输出文件名，添加时间戳
    if args.output_file:
        base_name, extension = os.path.splitext(args.output_file)
        args.output_file = f"{base_name}_{timestamp}{extension}"
    
    if args.json_output:
        base_name, extension = os.path.splitext(args.json_output)
        args.json_output = f"{base_name}_{timestamp}{extension}"
    
    print(f"Using output file: {args.output_file}")
    print(f"Using JSON output file: {args.json_output}")
    
    # 检查是否有continue_file并确定模式
    processed_videos = set()
    continue_mode = False
    
    if args.continue_file and os.path.exists(args.continue_file):
        continue_mode = True
        print(f"Continue mode: will load processed videos from {args.continue_file}")
        try:
            with open(args.continue_file, 'r') as f:
                existing_results = json.load(f)
                # 从数组格式中提取已处理的视频
                processed_videos = set(item.get('Video_name', '') for item in existing_results)
                print(f"Found {len(processed_videos)} already processed videos")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process continue file: {str(e)}")
            processed_videos = set()
            continue_mode = False
    else:
        print("Resume mode: will process all videos")
    
    # 初始化输出文件
    with open(args.output_file, "w") as f:
        pass
    
    # 初始化JSON输出文件（仅当文件不存在时）
    if not os.path.exists(args.json_output):
        os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump([], f)
        print(f"Initialized empty JSON output file: {args.json_output}")
    
    try:
        model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True, 
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        few_shot_examples = load_few_shot_examples(args.few_shot_dir, args.ground_truth_file, transform)
            
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        return

    with open(args.output_file, "a") as output_file:
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} test videos")
        
        for video_file in tqdm(video_files, desc="Processing test videos"):
            if continue_mode and video_file in processed_videos:
                print(f"Skipping already processed video: {video_file}")
                continue
            try:
                print(f"\nStarting to process: {video_file}")
                video_path = os.path.join(args.video_dir, video_file)
                
                print("Loading test video frames...")
                test_frames, test_num_patches = load_video(
                    video_path,
                    num_segments=args.num_frames,
                    max_num=1
                )
                test_frames = test_frames.to(torch.bfloat16).cuda()
                
                few_shot_prompt = generate_few_shot_prompt(few_shot_examples)
                conversation = generate_conversation(few_shot_prompt)
                
                print("Starting conversation...")
                # 修改这里：调用更新后的test_multi_turn_with_few_shot函数，传入Best-of-N样本数
                responses, evaluation_results = test_multi_turn_with_few_shot(
                    model, tokenizer, test_frames, test_num_patches,
                    few_shot_examples, conversation, device, video_file,
                    bon_samples=args.bon_samples
                )
                print("Conversation completed")
                
                # 保存对话日志
                with open(args.output_file, "a") as f:
                    f.write(f"\nProcessing video: {video_path}\n\n")
                    for i, (response, conv) in enumerate(zip(responses, conversation)):
                        f.write(f"Question: {conv}\n")
                        f.write(f"Answer: {response}\n")
                        f.write("-" * 20 + "\n")
                        
                # 在保存JSON结果部分，使用以下代码将结果追加到新文件中
                all_results = []
                # 如果continue_file存在且有效，则从continue_file中加载现有结果
                if continue_mode and args.continue_file and os.path.exists(args.continue_file):
                    try:
                        with open(args.continue_file, 'r') as f:
                            all_results = json.load(f)
                    except json.JSONDecodeError:
                        print("Warning: Could not parse existing continue file")

                # 将新的评估结果添加到数组中
                for metric, result in evaluation_results.items():
                    all_results.append(result)

                # 保存更新后的结果数组到新的输出文件
                with open(args.json_output, "a") as f:
                    json.dump(all_results, f, indent=2)
                
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory while processing {video_file}. Skipping...")
                output_file.write(f"\nError processing video {video_path}: CUDA out of memory\n\n")
                torch.cuda.empty_cache()
                continue
                
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                output_file.write(f"\nError processing video {video_path}: {str(e)}\n\n")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main()