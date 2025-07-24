"""
# 标准执行（从头开始，不继续之前的任务）
python test_gemma3.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "google/gemma-3-27b-it" \
  --output_file responses/output.txt \
  --json_output responses/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 

# 继续执行（继续处理之前未完成的视频）
python test_gemma3.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "google/gemma-3-27b-it" \
  --output_file results/output.txt \
  --json_output results/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --continue_file results/evaluation_20240508_143022.json
"""

import os
import torch 
import numpy as np
import argparse
import json
import re
import datetime
import traceback
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

IMAGENET_MEAN = (0.485, 0.456, 0.406) 
IMAGENET_STD = (0.229, 0.224, 0.225)


class FewShotExample:
    def __init__(self, image_path, ground_truth):
        self.image_path = image_path
        self.ground_truth = ground_truth
        self.processed_image = None

    def load_image(self):
        """Load the image using PIL."""
        self.processed_image = Image.open(self.image_path).convert('RGB')
        return self.processed_image


def parse_args():
    parser = argparse.ArgumentParser(description="Test Gemma-3 model with image-based few-shot learning")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing test video files")
    parser.add_argument("--few_shot_dir", type=str, required=True, help="Directory containing few-shot example images")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="JSON file containing ground truth for few-shot examples")
    parser.add_argument("--output_file", type=str, default="output.txt", help="File to save complete conversation logs")
    parser.add_argument("--json_output", type=str, default="evaluation.json", help="JSON file to save evaluation metrics")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it", help="Model name or path")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample from test videos")
    parser.add_argument("--continue_file", type=str, default=None, 
                       help="JSON file from previous run to continue processing from. If provided and exists, will skip processed videos.")
    parser.add_argument("--bon_samples", type=int, default=5, 
                       help="Number of samples to generate for Best-of-N strategy")
    return parser.parse_args()


def load_few_shot_examples(few_shot_dir, ground_truth_file):
    """Load few-shot examples from directory and ground truth file."""
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
            examples.append(example)
        else:
            print(f"Warning: No ground truth found for {image_file}")

    # Load all images for the examples
    for example in examples:
        example.load_image()
    
    return examples


def generate_few_shot_prompt(examples):
    """Generate few-shot prompt with example images."""
    prompt = "Here are some example AR screenshots with their quality assessment:\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Description:\n{example.ground_truth}\n\n"
    
    prompt += "Now, please understand the context of your task, and fully understand examples provided. Next, I will ask questions and please evaluate the AR effect in the Test Video.\n"
    return prompt


def select_best_response(responses, json_data):
    """Select best response from multiple samples using majority voting."""
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
    
    # Get majority vote
    majority_issue = "true" if issue_counts.get("true", 0) > issue_counts.get("false", 0) else "false"
    
    # Find the first response that matches the majority vote
    for i, json_obj in enumerate(valid_json):
        if str(json_obj["Issue"]).lower() == majority_issue:
            return valid_responses[i], json_obj
    
    # Default to first valid response
    return valid_responses[0], valid_json[0]

def check_tensor_alignment(tensor_dict):
    """Check if tensors are properly aligned in memory."""
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, stride={v.stride()}, alignment={v.storage_offset() % 16}")


def test_multi_turn_with_gemma3(model, processor, test_frames, few_shot_examples, conversation, device, video_file, bon_samples=5):
    """Run multi-turn conversation with Gemma3 model using few-shot learning and test time scaling."""
    # 基本内存管理
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 显式指定设备，避免可能的设备切换
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 确保只使用一个GPU
    
    responses = []
    evaluation_results = {}
    
    # 创建系统消息
    system_prompt = generate_system_prompt()
    few_shot_prompt = generate_few_shot_prompt(few_shot_examples)
    full_system_prompt = system_prompt + "\n\n" + few_shot_prompt
    
    # 初始化对话历史
    history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": full_system_prompt}]
        }
    ]
    
    # 初始验证问题
    for turn in range(2):  # 前两个验证问题
        # 创建带有视频帧的消息
        message_content = [{"type": "text", "text": conversation[turn]}]
        
        # 只在第一个问题添加图像
        if turn == 0:
            # 添加few-shot示例图像
            for i, example in enumerate(few_shot_examples):
                message_content.insert(0, {"type": "image", "image": example.processed_image})
                
            # 添加测试帧
            for frame in test_frames:
                message_content.append({"type": "image", "image": frame})
        
        # 添加消息到历史
        history.append(
            {
                "role": "user", 
                "content": message_content
            }
        )
        
        try:
            # 应用聊天模板
            inputs = processor.apply_chat_template(
                history, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            )
            
            # 首先将所有张量移到CPU，进行精确控制
            cpu_inputs = {k: v.cpu() for k, v in inputs.items() if torch.is_tensor(v)}
            
            # 然后明确分别处理不同类型的张量
            gpu_inputs = {}
            for k, v in cpu_inputs.items():
                if k == 'input_ids' or k == 'attention_mask':
                    # 整数类型张量保持为long类型，确保对齐和连续性
                    # 先进行连续操作，再移到GPU
                    v_cont = v.contiguous()
                    gpu_inputs[k] = v_cont.to(device=model.device, dtype=torch.long, non_blocking=False)
                else:
                    # 其他张量类型，确保对齐和连续性
                    v_cont = v.contiguous()
                    gpu_inputs[k] = v_cont.to(device=model.device, dtype=torch.bfloat16, non_blocking=False)
            
            # 确保张量同步，避免异步错误
            torch.cuda.synchronize()
            
            input_len = gpu_inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                # 使用try-except包装生成过程
                try:
                    generation = model.generate(
                        **gpu_inputs, 
                        max_new_tokens=512,
                        do_sample=False
                    )
                    generation = generation[0][input_len:]
                except RuntimeError as gen_error:
                    print(f"Generation error in turn {turn}: {str(gen_error)}")
                    # 尝试使用更保守的设置
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # 尝试更保守的生成选项
                    generation = model.generate(
                        **gpu_inputs, 
                        max_new_tokens=128,  # 减少token数量
                        do_sample=False
                    )
                    generation = generation[0][input_len:]
            
            # 确保GPU操作完成
            torch.cuda.synchronize()
            
            # 清理中间张量
            del cpu_inputs
            del gpu_inputs
            torch.cuda.empty_cache()
            
            # 解码响应
            response = processor.decode(generation, skip_special_tokens=True)
            responses.append(response)
            
            # 添加助手响应到历史
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })
            
            # 清理生成张量
            del generation
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error during turn {turn}: {str(e)}")
            traceback.print_exc()
            
            # 提供默认响应
            default_response = f"Could not process this question due to a technical error."
            responses.append(default_response)
            
            # 添加到历史以便继续对话
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": default_response}]
            })
    
    # 保存初始历史
    initial_history = history.copy()
    
    # 评估问题使用Best-of-N策略
    evaluation_responses = []
    metrics = ["Object Placement", "Object Movement", "Occlusion", 
               "Lighting", "Visual Artifacts", "Black Screen"]
    
    # 逐个处理评估问题
    for i in range(len(metrics)):
        try:
            # 重置历史到验证后
            current_history = initial_history.copy()
            
            # 获取带有详细指标描述的问题
            question_idx = i + 2  # 验证问题的偏移量为2
            question_with_metrics = conversation[question_idx]
            
            # 使用Best-of-N策略生成多个样本
            bon_responses = []
            bon_json_data = []
            
            for n in range(bon_samples):
                try:
                    # 创建带有评估问题的消息
                    message_content = [{"type": "text", "text": question_with_metrics}]
                    bon_history = current_history.copy()
                    bon_history.append({
                        "role": "user",
                        "content": message_content
                    })
                    
                    # 获取此样本的多样化参数
                    params = get_diverse_params(n, bon_samples)
                    
                    # 处理消息
                    inputs = processor.apply_chat_template(
                        bon_history, 
                        add_generation_prompt=True, 
                        tokenize=True,
                        return_dict=True, 
                        return_tensors="pt"
                    )
                    
                    # 首先将所有张量移到CPU，进行精确控制
                    cpu_inputs = {k: v.cpu() for k, v in inputs.items() if torch.is_tensor(v)}
                    
                    # 然后明确分别处理不同类型的张量
                    gpu_inputs = {}
                    for k, v in cpu_inputs.items():
                        if k == 'input_ids' or k == 'attention_mask':
                            # 先进行连续操作，再移到GPU
                            v_cont = v.contiguous()
                            gpu_inputs[k] = v_cont.to(device=model.device, dtype=torch.long, non_blocking=False)
                        else:
                            v_cont = v.contiguous()
                            gpu_inputs[k] = v_cont.to(device=model.device, dtype=torch.bfloat16, non_blocking=False)
                    
                    # 确保张量同步
                    torch.cuda.synchronize()
                    
                    input_len = gpu_inputs["input_ids"].shape[-1]
                    
                    # 使用try-except包装生成过程
                    with torch.inference_mode():
                        try:
                            generation = model.generate(
                                **gpu_inputs, 
                                max_new_tokens=min(512, params["max_new_tokens"]),
                                do_sample=params["do_sample"],
                                temperature=params["temperature"],
                                top_p=params["top_p"],
                                top_k=params["top_k"]
                            )
                            generation = generation[0][input_len:]
                        except RuntimeError as gen_error:
                            print(f"Generation error in sample {n} of metric {metrics[i]}: {str(gen_error)}")
                            # 尝试更保守的设置
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            
                            generation = model.generate(
                                **gpu_inputs, 
                                max_new_tokens=128,
                                do_sample=False,  # 禁用采样
                                temperature=1.0,
                                top_p=1.0
                            )
                            generation = generation[0][input_len:]
                    
                    # 确保GPU操作完成
                    torch.cuda.synchronize()
                    
                    # 清理中间张量
                    del cpu_inputs
                    del gpu_inputs
                    torch.cuda.empty_cache()
                    
                    # 解码响应
                    response = processor.decode(generation, skip_special_tokens=True)
                    bon_responses.append(response)
                    
                    # 尝试提取JSON
                    try:
                        json_content = extract_json_content(response)
                        if json_content:
                            json_data = json.loads(json_content)
                            json_data['Video_name'] = video_file
                            if 'Metrics' not in json_data:
                                json_data['Metrics'] = metrics[i]
                            bon_json_data.append(json_data)
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON for sample {n} of metric {metrics[i]}")
                    
                    # 清理生成张量
                    del generation
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error during sample {n} of metric {metrics[i]}: {str(e)}")
                    torch.cuda.empty_cache()  # 确保内存清理
            
            # 如果我们有有效响应，选择最佳响应
            if bon_responses:
                best_response, best_json = select_best_response(bon_responses, bon_json_data)
                evaluation_responses.append(best_response)
                
                # 存储结果
                if best_json:
                    evaluation_results[metrics[i]] = best_json
                else:
                    # 如果没有找到有效JSON，创建一个fallback结果
                    evaluation_results[metrics[i]] = {
                        "Video_name": video_file,
                        "Metrics": metrics[i],
                        "Issue": False,  # 默认值
                        "Reason": "Could not extract valid JSON from model output",
                        "error": "No valid JSON found"
                    }
            else:
                # 如果没有响应，创建一个默认响应
                default_response = f"Could not generate a valid response for {metrics[i]} evaluation."
                evaluation_responses.append(default_response)
                
                evaluation_results[metrics[i]] = {
                    "Video_name": video_file,
                    "Metrics": metrics[i],
                    "Issue": False,  # 默认值
                    "Reason": "Could not generate due to technical limitations",
                    "error": "Generation failed"
                }
        
        except Exception as e:
            print(f"Exception during metric {metrics[i]}: {str(e)}")
            traceback.print_exc()
            
            # 添加默认响应和结果
            default_response = f"Failed to evaluate {metrics[i]} due to an error."
            evaluation_responses.append(default_response)
            
            evaluation_results[metrics[i]] = {
                "Video_name": video_file,
                "Metrics": metrics[i],
                "Issue": False,
                "Reason": "Evaluation failed due to technical error",
                "error": "Exception occurred"
            }
            
            # 清理可能的内存残留
            torch.cuda.empty_cache()
    
    # 确保我们有足够的响应
    while len(evaluation_responses) < 6:
        evaluation_responses.append("No response generated")
    
    # 最后关于其他问题的询问
    try:
        final_history = initial_history.copy()
        
        # 添加所有评估问题和响应到历史
        for i in range(min(6, len(evaluation_responses))):
            final_history.append({
                "role": "user",
                "content": [{"type": "text", "text": conversation[i+2]}]
            })
            final_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": evaluation_responses[i]}]
            })
        
        # 添加最后的问题
        final_history.append({
            "role": "user",
            "content": [{"type": "text", "text": conversation[8]}]
        })
        
        # 处理最后的问题
        inputs = processor.apply_chat_template(
            final_history, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # 首先将所有张量移到CPU，进行精确控制
        cpu_inputs = {k: v.cpu() for k, v in inputs.items() if torch.is_tensor(v)}
        
        # 然后明确分别处理不同类型的张量
        gpu_inputs = {}
        for k, v in cpu_inputs.items():
            if k == 'input_ids' or k == 'attention_mask':
                v_cont = v.contiguous()
                gpu_inputs[k] = v_cont.to(device=model.device, dtype=torch.long, non_blocking=False)
            else:
                v_cont = v.contiguous()
                gpu_inputs[k] = v_cont.to(device=model.device, dtype=torch.bfloat16, non_blocking=False)
        
        # 确保张量同步
        torch.cuda.synchronize()
        
        input_len = gpu_inputs["input_ids"].shape[-1]
        
        # 使用try-except包装生成过程
        with torch.inference_mode():
            try:
                generation = model.generate(
                    **gpu_inputs, 
                    max_new_tokens=256,
                    do_sample=False
                )
                generation = generation[0][input_len:]
            except RuntimeError as gen_error:
                print(f"Generation error in final question: {str(gen_error)}")
                # 尝试更保守的设置
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                generation = model.generate(
                    **gpu_inputs, 
                    max_new_tokens=128,
                    do_sample=False
                )
                generation = generation[0][input_len:]
        
        # 确保GPU操作完成
        torch.cuda.synchronize()
        
        # 清理中间张量
        del cpu_inputs
        del gpu_inputs
        torch.cuda.empty_cache()
        
        # 解码最后的响应
        final_response = processor.decode(generation, skip_special_tokens=True)
        
        # 清理生成张量
        del generation
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Exception during final question: {str(e)}")
        traceback.print_exc()
        final_response = "Sorry, I couldn't process the final question due to a technical issue."
    
    # 收集所有响应
    responses.extend(evaluation_responses)
    responses.append(final_response)
    
    return responses, evaluation_results


def save_json_results(file_path, video_file, evaluation_results, continue_file=None):
    """Save JSON results to file, ensuring existing data is preserved."""
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
    
    # Write all results to file
    with open(file_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved {len(all_results)} total results to {file_path}")
    return all_results


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


def generate_system_prompt():
    """Generate system prompt for AR evaluation."""
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


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Get frame indices for video sampling."""
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


def load_video(video_path, bound=None, num_segments=32):
    """Load video frames from path."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    frames = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        frames.append(img)
    
    return frames


def generate_conversation():
    """Generate conversation structure for AR evaluation."""
    conversation = [
        "Based on the sampled frame I uploaded from the video. What do you see in this video? Can you see the AR effect in the video? You don't need to do the evaluation here now.",
        "What is the 3D object in the video? If you can see the 3D object, only answer what the object is without description. If you are not clear what exactly the object is, try to describe it.",
        "Is there any issue with the Object Placement? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        "Is there any issue with the Occlusion? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        "Is there any issue with the Object Movement? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.", 
        "Is there any issue with the Lighting? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        "Is there any issue with Visual Artifacts and Rendering Issues? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        "Is there any issue of Black Screen? Please provide your answer in JSON format with the following fields: 'Video_name', 'Metrics', 'Issue' (boolean), and 'Reason'.",
        "Except Object Placement, Object Movement, Occlusion, Lighting, Artifacts and Rendering and Black Screen, have you found any other issues about Augmented Reality in this video?"
    ]
    
    return conversation


def get_diverse_params(sample_index, total_samples):
    """Generate diverse parameters for Best-of-N strategy."""
    temperature = 0.5 + (0.5 * sample_index / total_samples)  # Temperature from 0.5 to 1.0
    top_p = 0.85 + (0.1 * sample_index / total_samples)  # Top-p from 0.85 to 0.95
    
    return {
        "temperature": temperature,
        "do_sample": True,
        "top_p": top_p,
        "top_k": 40,
        "max_new_tokens": 2048
    }


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

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped output filenames
    if args.output_file:
        base_name, extension = os.path.splitext(args.output_file)
        args.output_file = f"{base_name}_{timestamp}{extension}"
    
    if args.json_output:
        base_name, extension = os.path.splitext(args.json_output)
        args.json_output = f"{base_name}_{timestamp}{extension}"
    
    print(f"Using output file: {args.output_file}")
    print(f"Using JSON output file: {args.json_output}")
    
    # Check for continue mode
    processed_videos = set()
    continue_mode = False
    
    if args.continue_file and os.path.exists(args.continue_file):
        continue_mode = True
        print(f"Continue mode: will load processed videos from {args.continue_file}")
        try:
            with open(args.continue_file, 'r') as f:
                existing_results = json.load(f)
                # Extract processed videos from results
                processed_videos = set(item.get('Video_name', '') for item in existing_results)
                print(f"Found {len(processed_videos)} already processed videos")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process continue file: {str(e)}")
            processed_videos = set()
            continue_mode = False
    else:
        print("Resume mode: will process all videos")
    
    # Initialize output files
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        pass
    
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    with open(args.json_output, "w") as f:
        json.dump([], f)
    
    try:
        # Load model and processor
        print(f"Loading Gemma3 model: {args.model_name}")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_name, 
            device_map="auto"
        ).eval()
        
        processor = AutoProcessor.from_pretrained(args.model_name)
        
        # Load few-shot examples
        print(f"Loading few-shot examples from {args.few_shot_dir}")
        few_shot_examples = load_few_shot_examples(args.few_shot_dir, args.ground_truth_file)
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
        
        # Generate conversation structure
        conversation = generate_conversation()
        
        # Process videos
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} test videos")
        
        for video_file in tqdm(video_files, desc="Processing test videos"):
            if continue_mode and video_file in processed_videos:
                print(f"Skipping already processed video: {video_file}")
                continue
                
            try:
                print(f"\nStarting to process: {video_file}")
                video_path = os.path.join(args.video_dir, video_file)
                
                # Load test video frames
                print("Loading test video frames...")
                test_frames = load_video(video_path, num_segments=args.num_frames)
                print(f"Loaded {len(test_frames)} frames")
                
                # Run multi-turn conversation
                print(f"Starting conversation with Best-of-{args.bon_samples} strategy...")
                responses, evaluation_results = test_multi_turn_with_gemma3(
                    model, processor, test_frames, few_shot_examples, 
                    conversation, device, video_file, bon_samples=args.bon_samples
                )
                print("Conversation completed")
                
                # Save conversation log
                with open(args.output_file, "a") as f:
                    f.write(f"\nProcessing video: {video_path}\n\n")
                    for i, response in enumerate(responses):
                        question = conversation[i] if i < len(conversation) else "Additional question"
                        f.write(f"Question: {question}\n")
                        f.write(f"Answer: {response}\n")
                        f.write("-" * 20 + "\n")
                
                # Save JSON results
                save_json_results(
                    args.json_output,
                    video_file,
                    evaluation_results,
                    continue_file=args.continue_file if continue_mode else None
                )
                
                # Clear CUDA cache to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory while processing {video_file}. Skipping...")
                with open(args.output_file, "a") as f:
                    f.write(f"\nError processing video {video_path}: CUDA out of memory\n\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
                
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                traceback.print_exc()
                with open(args.output_file, "a") as f:
                    f.write(f"\nError processing video {video_path}: {str(e)}\n\n")
                continue
    
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()