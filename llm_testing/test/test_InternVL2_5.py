"""
# Standard execution (start fresh)
python test_InternVL2_5.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "OpenGVLab/InternVL2_5-8B-MPO" \
  --output_file responses/output.txt \
  --json_output responses/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --use_fewshot True

# Continue execution (resume from where it left off)
python test_InternVL2_5.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "OpenGVLab/InternVL2_5-8B-MPO" \
  --output_file results/output.txt \
  --json_output results/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --continue_file results/evaluation_20240508_143022.json \
  --use_fewshot False
"""

import os
import torch 
import numpy as np
import argparse
import math
import json
import re
import datetime
import traceback
import time
import gc
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Import AR evaluation modules
from ar_prompts import (
    generate_system_prompt, 
    generate_few_shot_prompt, 
    generate_conversation_questions,
    generate_ar_metrics_description,
    METRICS
)
from ar_evaluation import (
    load_video_decord, 
    get_frame_indices,
    extract_json_content, 
    save_json_results,
    get_diverse_params, 
    select_best_response,
    generate_timestamp_filename,
    get_processed_videos,
    format_time
)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Test InternVL2_5 model with image-based few-shot learning")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing test video files")
    parser.add_argument("--few_shot_dir", type=str, default=None, help="Directory containing few-shot example images")
    parser.add_argument("--ground_truth_file", type=str, default=None, help="JSON file containing ground truth for few-shot examples")
    parser.add_argument("--output_file", type=str, default="output.txt", help="File to save complete conversation logs")
    parser.add_argument("--json_output", type=str, default="evaluation.json", help="JSON file to save evaluation metrics")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL2_5-8B", help="Model name or path")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to sample from test videos")
    parser.add_argument("--continue_file", type=str, default=None, 
                      help="JSON file from previous run to continue processing from. If provided and exists, will skip processed videos.")
    parser.add_argument("--bon_samples", type=int, default=16, 
                      help="Number of samples to generate for Best-of-N strategy")
    parser.add_argument("--use_fewshot", type=bool, default=False,
                      help="Whether to use few-shot examples for improved performance")

    return parser.parse_args()


def load_few_shot_examples(few_shot_dir, ground_truth_file, transform):
    """Load few-shot examples with better error handling."""
    if not few_shot_dir or not ground_truth_file:
        print("Few-shot learning disabled or missing directories")
        return []
        
    if not os.path.exists(few_shot_dir):
        print(f"Few-shot directory not found: {few_shot_dir}")
        return []
        
    if not os.path.exists(ground_truth_file):
        print(f"Ground truth file not found: {ground_truth_file}")
        return []
    
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {ground_truth_file}")
        return []
    except Exception as e:
        print(f"Error loading ground truth file: {str(e)}")
        return []
    
    examples = []
    image_files = [f for f in os.listdir(few_shot_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Found {len(image_files)} few-shot example images in {few_shot_dir}")
    
    for image_file in image_files:
        image_path = os.path.join(few_shot_dir, image_file)
        if image_file in ground_truth_data:
            try:
                example = FewShotExample(
                    image_path=image_path,
                    ground_truth=ground_truth_data[image_file]
                )
                example.load_image(transform)
                examples.append(example)
            except Exception as e:
                print(f"Error loading few-shot example {image_file}: {str(e)}")
        else:
            print(f"Warning: No ground truth found for {image_file}")
    
    print(f"Successfully loaded {len(examples)} few-shot examples")
    return examples


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

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_frame_indices(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
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
        print("No GPUs found, using CPU.")
    return device


def check_tensor_alignment(tensor_dict):
    """Check if tensors are properly aligned in memory."""
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={v.shape}, stride={v.stride()}, alignment={v.storage_offset() % 16}")


def clean_gpu_memory():
    """Actively clean GPU memory to prevent OOM errors."""
    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Perform garbage collection
        gc.collect()
        
        # Optional: print memory stats for debugging
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
                print(f"GPU {i}: Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


def test_multi_turn_with_few_shot(model, tokenizer, test_frames, test_num_patches, few_shot_examples, conversation, device, video_file, bon_samples=16):
    """
    Run multi-turn conversation with InternVL2.5 model using few-shot learning
    and Best-of-N strategy for improved response quality.
    """
    model.eval()
    responses = []
    evaluation_results = {}
    generation_config = dict(max_new_tokens=1024, do_sample=True, top_k=40, top_p=0.95, temperature=0.7)
    
    # Combine example frames and test video frames
    all_frames = []
    all_num_patches = []
    
    # Add few-shot example frames if available
    if few_shot_examples:
        for example in few_shot_examples:
            all_frames.append(example.processed_image)
            all_num_patches.extend(example.num_patches)
    
    # Add test frames
    all_frames.append(test_frames)
    all_num_patches.extend(test_num_patches)
    combined_frames = torch.cat(all_frames)
    
    # Initial verification questions (context and first two questions)
    history = None
    for turn in range(3):  # Context and first two verification questions
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
    
    # Save initial history record
    initial_history = history
    
    # Six evaluation questions (using Best-of-N strategy)
    evaluation_responses = []
    
    for i in range(6):  # Six evaluation questions
        history = initial_history  # Reset history to after verification questions
        question = conversation[i + 3]  # Evaluation questions offset by 3
        
        # Use Best-of-N strategy to generate multiple answers
        bon_responses = []
        bon_json_data = []
        
        # Generate N samples
        for n in range(bon_samples):
            try:
                # Get diverse parameters for this sample
                params = get_diverse_params(n, bon_samples)
                
                # Generate response
                response, _ = model.chat(
                    tokenizer,
                    combined_frames,
                    question,
                    params,  # Use diverse parameters
                    num_patches_list=all_num_patches,
                    history=history,
                    return_history=True
                )
                bon_responses.append(response)
                
                # Try to parse JSON response
                try:
                    json_content = extract_json_content(response)
                    if json_content:
                        json_data = json.loads(json_content)
                        json_data['Video_name'] = video_file
                        if 'Metrics' not in json_data:
                            json_data['Metrics'] = METRICS[i]
                        bon_json_data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for sample {n+1}/{bon_samples} of metric {METRICS[i]}")
                
                # Clean GPU memory after each sample generation
                # This is safe because we're not building on previous outputs
                # and have saved all necessary information
                clean_gpu_memory()
                
            except Exception as e:
                print(f"Error generating sample {n+1}/{bon_samples} for metric {METRICS[i]}: {str(e)}")
                # Still clean memory even if there was an error
                clean_gpu_memory()
        
        # Select best answer (using majority vote)
        best_response, best_json = select_best_response(bon_responses, bon_json_data)
        evaluation_responses.append(best_response)
        
        if best_json:
            evaluation_results[METRICS[i]] = best_json
        else:
            # If no valid JSON found, create a default result
            evaluation_results[METRICS[i]] = {
                "Video_name": video_file,
                "Metrics": METRICS[i],
                "Issue": False,  # Default to no issue
                "Reason": "Could not determine from model output",
                "error": "No valid JSON found"
            }
        
        # Clean up memory after processing each metric
        clean_gpu_memory()
    
    # Final extended question (not using BoN)
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


def main():
    # Record program start time
    program_start_time = time.time()
    
    args = parse_args()
    device = setup_device()
    
    # Extract scale from model name
    scale = args.model_name.split("-")[1]
    device_map = split_model(f'InternVL2_5-{scale}')
    transform = build_transform(input_size=448)
    
    # Generate filename with timestamp
    if args.output_file:
        args.output_file = generate_timestamp_filename(args.output_file)
    
    if args.json_output:
        args.json_output = generate_timestamp_filename(args.json_output)
    
    print(f"Using output file: {args.output_file}")
    print(f"Using JSON output file: {args.json_output}")
    print(f"Using few-shot learning: {args.use_fewshot}")
    
    # Check if continue_file exists and determine mode
    processed_videos = set()
    continue_mode = False
    
    if args.continue_file and os.path.exists(args.continue_file):
        continue_mode = True
        print(f"Continue mode: will load processed videos from {args.continue_file}")
        processed_videos = get_processed_videos(args.continue_file)
        print(f"Found {len(processed_videos)} already processed videos")
    else:
        print("Starting fresh: will process all videos")
    
    # Initialize output files
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        pass
    
    # Initialize JSON output file
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    with open(args.json_output, "w") as f:
        json.dump([], f)
    
    try:
        print(f"Loading model {args.model_name}...")
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
        
        # Load few-shot examples if enabled
        few_shot_examples = []
        if args.use_fewshot:
            print("Loading few-shot examples...")
            few_shot_examples = load_few_shot_examples(args.few_shot_dir, args.ground_truth_file, transform)
            if not few_shot_examples:
                print("Warning: Few-shot learning enabled but no examples were loaded. Continuing without few-shot examples.")
        
        # Generate conversation structure
        conversation = generate_conversation_questions(include_descriptions=True)
        
        # Add system prompt and few-shot prompt if enabled
        if args.use_fewshot and few_shot_examples:
            few_shot_prompt = generate_few_shot_prompt(few_shot_examples)
            full_system_prompt = generate_system_prompt() + "\n\n" + few_shot_prompt
        else:
            full_system_prompt = generate_system_prompt()
            
        conversation.insert(0, full_system_prompt)
            
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        return

    # Process video files
    with open(args.output_file, "a") as output_file:
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} test videos")
        
        successful_videos = 0
        failed_videos = 0
        skipped_videos = 0
        
        for video_file in tqdm(video_files, desc="Processing test videos"):
            if continue_mode and video_file in processed_videos:
                print(f"Skipping already processed video: {video_file}")
                skipped_videos += 1
                continue
                
            try:
                print(f"\nStarting to process: {video_file}")
                video_path = os.path.join(args.video_dir, video_file)
                
                # Load video frames
                print("Loading test video frames...")
                test_frames, test_num_patches = load_video(
                    video_path,
                    num_segments=args.num_frames,
                    max_num=1
                )
                test_frames = test_frames.to(torch.bfloat16).cuda()
                
                # Run multi-turn conversation
                print(f"Starting conversation with Best-of-{args.bon_samples} strategy...")
                responses, evaluation_results = test_multi_turn_with_few_shot(
                    model, tokenizer, test_frames, test_num_patches,
                    few_shot_examples, conversation, device, video_file,
                    bon_samples=args.bon_samples
                )
                print("Conversation completed")
                
                # Save conversation log
                with open(args.output_file, "a") as f:
                    f.write(f"\nProcessing video: {video_path}\n\n")
                    for i, (response, conv) in enumerate(zip(responses, conversation)):
                        # Skip first system prompt
                        if i == 0:
                            continue
                        # For remaining content, use index-1 to match question
                        question_idx = i - 1
                        f.write(f"Question: {conversation[question_idx+1]}\n")
                        f.write(f"Answer: {response}\n")
                        f.write("-" * 20 + "\n")
                
                # Save JSON results
                save_json_results(
                    args.json_output,
                    video_file,
                    evaluation_results,
                    continue_file=args.continue_file if continue_mode else None
                )
                
                # Free GPU memory
                clean_gpu_memory()
                
                successful_videos += 1
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory while processing {video_file}. Skipping...")
                output_file.write(f"\nError processing video {video_path}: CUDA out of memory\n\n")
                clean_gpu_memory()
                failed_videos += 1
                continue
                
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                output_file.write(f"\nError processing video {video_path}: {str(e)}\n\n")
                traceback.print_exc()
                failed_videos += 1
                continue
    
    # Calculate total runtime
    program_end_time = time.time()
    total_elapsed = program_end_time - program_start_time
    
    # Output execution summary
    print("\n" + "="*50)
    print("Execution Summary:")
    print(f"Total time: {format_time(total_elapsed)}")
    print(f"Videos: {successful_videos} successful, {failed_videos} failed, {skipped_videos} skipped")
    print("="*50)


if __name__ == "__main__":
    main()