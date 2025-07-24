"""
# 标准执行（从头开始，不继续之前的任务）
python test_mistral_optimized.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
  --output_file responses/output.txt \
  --json_output responses/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --quantize 4bit \
  --device_map "balanced" \
  --offload_images

# 继续执行（继续处理之前未完成的视频）
python test_mistral_optimized.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
  --output_file responses/output.txt \
  --json_output responses/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --continue_file responses/evaluation_20240508_143022.json \
  --quantize 4bit \
  --device_map "balanced" \
  --offload_images
"""

import os
import torch
import argparse
import json
import traceback
import time
import gc
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# 导入自定义模块
import ar_prompts
import ar_evaluation

class FewShotExample:
    """用于存储和处理少样本示例的类"""
    def __init__(self, image_path, ground_truth):
        self.image_path = image_path
        self.ground_truth = ground_truth
        self.image = None
        self._image_loaded = False
        
    def load_image(self):
        """加载图像并返回PIL格式，保持在CPU内存中"""
        if not self._image_loaded:
            try:
                # 尝试加载图像并确保它保持在CPU内存中
                self.image = Image.open(self.image_path).convert('RGB')
                self._image_loaded = True
            except Exception as e:
                print(f"Error loading image {self.image_path}: {str(e)}")
                raise
        return self.image


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test Mistral model for AR evaluation with optimizations")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing test video files")
    parser.add_argument("--few_shot_dir", type=str, required=True, help="Directory containing few-shot example images")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="JSON file containing ground truth for few-shot examples")
    parser.add_argument("--output_file", type=str, default="output.txt", help="File to save complete conversation logs")
    parser.add_argument("--json_output", type=str, default="evaluation.json", help="JSON file to save evaluation metrics")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help="Model name or path")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample from test videos")
    parser.add_argument("--continue_file", type=str, default=None, 
                       help="JSON file from previous run to continue processing from. If provided and exists, will skip processed videos.")
    parser.add_argument("--bon_samples", type=int, default=5, 
                       help="Number of samples to generate for Best-of-N strategy")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map for model distribution")
    parser.add_argument("--quantize", type=str, choices=["none", "4bit", "8bit"], default="4bit",
                       help="Quantization strategy to use (none, 4bit, or 8bit)")
    parser.add_argument("--offload_images", action="store_true", default=True,
                       help="Keep images in CPU memory and only load to GPU when needed")
    return parser.parse_args()


def load_few_shot_examples(few_shot_dir, ground_truth_file):
    """加载少样本示例和对应的ground truth数据"""
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
            example.load_image()  # 图像保持在CPU内存中
            examples.append(example)
        else:
            print(f"Warning: No ground truth found for {image_file}")
    
    print(f"Loaded {len(examples)} few-shot examples (keeping images in CPU memory)")
    return examples


def setup_model_and_processor(model_name, quantize="4bit", device_map="auto"):
    """设置模型和处理器，支持量化和设备映射
    
    Args:
        model_name: 模型名称或路径
        quantize: 量化策略 ("none", "4bit", or "8bit")
        device_map: 设备映射策略
        
    Returns:
        处理器和模型的元组
    """
    print(f"Loading Mistral model: {model_name}")
    
    # 首先加载处理器
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 配置模型加载参数
    model_kwargs = {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    
    # 根据量化选项配置
    if quantize == "4bit":
        print("Using 4-bit quantization to reduce memory usage")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = quantization_config
        
    elif quantize == "8bit":
        print("Using 8-bit quantization to reduce memory usage")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        print("Using full precision model (no quantization)")
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # 加载模型
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # 绑定权重以解决警告
    model.tie_weights()
    
    # 设置pad_token_id（如果未设置）
    if model.config.pad_token_id is None:
        if model.config.eos_token_id is not None:
            model.config.pad_token_id = model.config.eos_token_id
        else:
            # Mistral模型常用的默认pad_token_id
            model.config.pad_token_id = 2
    
    return processor, model


def prepare_images_for_model(images, processor, offload_images=True):
    """准备图像用于模型输入，支持CPU-GPU卸载
    
    Args:
        images: 图像列表
        processor: 模型处理器
        offload_images: 是否使用图像卸载（保持在CPU上，需要时加载到GPU）
        
    Returns:
        处理后的图像张量
    """
    if offload_images:
        # 首先确保图像在CPU上
        cpu_images = []
        for img in images:
            if hasattr(img, 'to') and callable(getattr(img, 'to')):
                cpu_images.append(img.to('cpu'))
            else:
                cpu_images.append(img)
        
        # 处理图像（保持在CPU上）
        with torch.no_grad():
            processed = processor.image_processor(cpu_images, return_tensors="pt")
        
        return processed
    else:
        # 直接处理图像
        with torch.no_grad():
            processed = processor.image_processor(images, return_tensors="pt")
        
        return processed
    
def track_gpu_memory(label="Current GPU memory usage"):
    """追踪当前GPU内存使用情况并返回可读格式
    
    Args:
        label: 标记用于日志输出
        
    Returns:
        包含GPU内存使用情况的字符串
    """
    if not torch.cuda.is_available():
        return f"{label}: No GPU available"
        
    memory_stats = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        free_memory = total_memory - reserved_memory
        
        memory_stats.append(
            f"GPU {i}: {allocated_memory:.2f}GB/{total_memory:.2f}GB "
            f"(Reserved: {reserved_memory:.2f}GB, Free: {free_memory:.2f}GB)"
        )
    
    memory_info = " | ".join(memory_stats)
    return f"{label}: {memory_info}"

def run_conversation(processor, model, test_frames, few_shot_examples, conversation, video_file, bon_samples=5, offload_images=True, track_memory=True):
    """执行与Mistral模型的多轮对话，支持图像卸载和GPU内存追踪
    
    Args:
        processor: 模型处理器
        model: Mistral模型
        test_frames: 测试视频帧列表
        few_shot_examples: 少样本示例列表
        conversation: 对话问题列表
        video_file: 视频文件名
        bon_samples: Best-of-N策略的样本数
        offload_images: 是否使用图像卸载（保持在CPU上，需要时加载到GPU）
        track_memory: 是否追踪GPU内存使用情况
    """
    # 初始内存追踪
    if track_memory and torch.cuda.is_available():
        print(track_gpu_memory("Initial GPU memory before conversation"))
    
    # 主动清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if track_memory:
            print(track_gpu_memory("GPU memory after initial cleanup"))
    
    responses = []
    evaluation_results = {}
    
    # 初始化对话历史
    history = []
    
    # 添加系统提示和示例
    system_prompt = ar_prompts.generate_system_prompt()
    few_shot_prompt = ar_prompts.generate_few_shot_prompt(few_shot_examples)
    
    # 开始对话的前两个验证问题
    for turn in range(2):
        if track_memory and torch.cuda.is_available():
            print(track_gpu_memory(f"GPU memory before turn {turn+1} verification question"))
        
        # 创建包含图像和文本的消息
        messages = []
        
        # 第一轮加入系统提示和所有图像，后续轮次只需添加文本
        if turn == 0:
            # 添加系统提示
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt + "\n\n" + few_shot_prompt}]
            })
            
            # 用户消息
            user_content = []
            
            # 准备图像（示例图像和测试帧）
            all_images = []
            
            # 添加示例图像
            for example in few_shot_examples:
                all_images.append(example.image)
            
            # 添加测试帧
            all_images.extend(test_frames)
            
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory before adding images"))
            
            # 处理所有图像
            if offload_images:
                # 如果使用图像卸载，只在需要时将图像加载到GPU
                # 先提取示例图像
                for i, example in enumerate(few_shot_examples):
                    user_content.append({"type": "image", "image": example.image})
                
                # 再添加测试帧
                for frame in test_frames:
                    user_content.append({"type": "image", "image": frame})
            else:
                # 直接处理和添加所有图像
                for image in all_images:
                    user_content.append({"type": "image", "image": image})
            
            # 添加文本提示
            user_content.append({"type": "text", "text": conversation[turn]})
            
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory after adding images"))
        else:
            # 添加历史对话
            messages.extend(history)
            
            # 添加新问题
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": conversation[turn]}]
            })
        
        try:
            # 手动垃圾回收和CUDA缓存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if track_memory:
                    print(track_gpu_memory("GPU memory after cleanup before processing"))
            
            # 处理输入
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory before tokenization"))
                
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            )
            
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory after tokenization, before moving to GPU"))
            
            # 将输入移至GPU
            inputs = inputs.to(model.device, dtype=torch.bfloat16)
            
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory after moving inputs to GPU"))
            
            # 生成回答
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory before generation"))
                
            with torch.inference_mode():
                generate_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    pad_token_id=model.config.pad_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95
                )
            
            if track_memory and torch.cuda.is_available():
                print(track_gpu_memory("GPU memory after generation"))
            
            # 解码输出
            response = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            responses.append(response)
            
            # 更新历史记录
            history.append({
                "role": "user",
                "content": [{"type": "text", "text": conversation[turn]}]
            })
            
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })
            
            # 清理内存
            del inputs
            del generate_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if track_memory:
                    print(track_gpu_memory("GPU memory after cleanup"))
                
        except Exception as e:
            print(f"Error in conversation turn {turn}: {str(e)}")
            # 如果出错，添加一个空响应以保持对话流程
            responses.append("I'm unable to respond to this question at the moment.")
            history.append({
                "role": "user",
                "content": [{"type": "text", "text": conversation[turn]}]
            })
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "I'm unable to respond to this question at the moment."}]
            })
    
    # 保存初始历史记录以用于评估问题
    initial_history = history.copy()
    
    # 处理6个评估问题（使用Best-of-N策略）
    evaluation_responses = []
    metrics = ar_prompts.METRICS
    
    for i in range(6):  # 六个评估问题
        # 主动清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if track_memory:
                print(track_gpu_memory(f"\nGPU memory before metric {i+1}: {metrics[i]}"))
            
        print(f"Evaluating metric: {metrics[i]}")
        
        # 重置历史记录到初始状态
        history = initial_history.copy()
        
        # 添加评估问题
        question = conversation[i + 2]
        
        # 使用Best-of-N策略获取多个样本，但将结果保存到CPU
        bon_responses = []
        bon_json_data = []
        
        for n in range(bon_samples):
            try:
                print(f"Generating sample {n+1}/{bon_samples} for metric {metrics[i]}...")
                
                if track_memory and torch.cuda.is_available():
                    print(track_gpu_memory(f"GPU memory before sample {n+1}/{bon_samples}"))
                
                # 构建消息历史
                messages = []
                messages.extend(history)
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": question}]
                })
                
                # 获取多样化参数
                params = ar_evaluation.get_diverse_params(n, bon_samples)
                
                # 手动垃圾回收和CUDA缓存清理
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if track_memory:
                        print(track_gpu_memory("GPU memory after cleanup before processing"))
                
                # 处理输入
                inputs = processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_dict=True, 
                    return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                
                if track_memory and torch.cuda.is_available():
                    print(track_gpu_memory("GPU memory after input preparation"))
                
                # 生成回答
                with torch.inference_mode():
                    generate_ids = model.generate(
                        **inputs, 
                        max_new_tokens=params["max_new_tokens"],
                        pad_token_id=model.config.pad_token_id,
                        do_sample=params["do_sample"],
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        top_k=params["top_k"]
                    )
                
                if track_memory and torch.cuda.is_available():
                    print(track_gpu_memory("GPU memory after generation"))
                
                # 解码输出
                response = processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # 将响应保存到CPU
                bon_responses.append(response)
                
                # 尝试提取JSON（在CPU上进行）
                try:
                    json_content = ar_evaluation.extract_json_content(response)
                    if json_content:
                        json_data = json.loads(json_content)
                        json_data['Video_name'] = video_file
                        if 'Metrics' not in json_data:
                            json_data['Metrics'] = metrics[i]
                        bon_json_data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for sample {n} of metric {metrics[i]}")
                
                # 显式删除不再需要的GPU张量并清理缓存
                del inputs
                del generate_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if track_memory:
                        print(track_gpu_memory(f"GPU memory after cleanup for sample {n+1}"))
                print(f"Memory freed after sample {n+1}")
                
            except Exception as e:
                print(f"Error generating sample {n+1} for metric {metrics[i]}: {str(e)}")
                traceback.print_exc()
                # 如果这个样本失败，继续尝试下一个
                continue
        
        # 在CPU上选择最佳响应（不需要GPU）
        best_response, best_json = ar_evaluation.select_best_response(bon_responses, bon_json_data)
        evaluation_responses.append(best_response)
        
        if best_json:
            evaluation_results[metrics[i]] = best_json
        else:
            # 如果没有有效的JSON，创建默认结果
            evaluation_results[metrics[i]] = {
                "Video_name": video_file,
                "Metrics": metrics[i],
                "Issue": False,  # 默认值
                "Reason": "Could not determine from model output",
                "error": "No valid JSON found"
            }
        
        # 更新历史记录（只添加最佳的响应到历史记录中）
        history.append({
            "role": "user",
            "content": [{"type": "text", "text": question}]
        })
        
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": best_response}]
        })
        
        # 完成一个指标后主动清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if track_memory:
                print(track_gpu_memory(f"GPU memory after completing metric {metrics[i]}"))
        print(f"Completed evaluation for metric: {metrics[i]}")
    
    # 处理最后一个问题（非评估问题）
    try:
        # 主动清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if track_memory:
                print(track_gpu_memory("\nGPU memory before final question"))
            
        final_question = conversation[8]  # 第9个问题
        
        # 构建消息历史
        messages = []
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": final_question}]
        })
        
        # 处理输入
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        # 生成回答
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                pad_token_id=model.config.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        
        # 解码输出
        final_response = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 收集所有响应
        responses.extend(evaluation_responses)
        responses.append(final_response)
        
        # 清理内存
        del inputs
        del generate_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if track_memory:
                print(track_gpu_memory("GPU memory after final question"))
        
    except Exception as e:
        print(f"Error in final question: {str(e)}")
        traceback.print_exc()
        final_response = "I'm unable to respond to this question at the moment."
        
        # 收集所有响应
        responses.extend(evaluation_responses)
        responses.append(final_response)
    
    # 最终内存追踪
    if track_memory and torch.cuda.is_available():
        print(track_gpu_memory("\nFinal GPU memory after complete conversation"))
    
    return responses, evaluation_results


def display_gpu_info():
    """显示GPU使用信息"""
    if torch.cuda.is_available():
        print("\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            free_memory = total_memory - reserved_memory
            
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Free memory: {free_memory:.2f} GB")
            print(f"  Reserved memory: {reserved_memory:.2f} GB")
            print(f"  Allocated memory: {allocated_memory:.2f} GB")
    else:
        print("No GPU available. Running on CPU.")


def main():
    # 记录程序开始时间
    start_time = time.time()
    
    args = parse_args()
    
    # 生成带时间戳的输出文件名
    args.output_file = ar_evaluation.generate_timestamp_filename(args.output_file)
    args.json_output = ar_evaluation.generate_timestamp_filename(args.json_output)
    
    print(f"Using output file: {args.output_file}")
    print(f"Using JSON output file: {args.json_output}")
    print(f"Image offloading: {'Enabled' if args.offload_images else 'Disabled'}")
    
    # 检查是否处于继续模式
    processed_videos = set()
    continue_mode = False
    
    if args.continue_file and os.path.exists(args.continue_file):
        continue_mode = True
        print(f"Continue mode: will load processed videos from {args.continue_file}")
        processed_videos = ar_evaluation.get_processed_videos(args.continue_file)
        print(f"Found {len(processed_videos)} already processed videos")
    else:
        print("New run: will process all videos")
    
    # 初始化输出文件
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        pass
    
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    with open(args.json_output, "w") as f:
        json.dump([], f)
    
    successful_videos = 0
    failed_videos = 0
    skipped_videos = 0
    
    try:
        # 初始化模型和处理器
        print(f"Loading Mistral model: {args.model_name}")
        processor, model = setup_model_and_processor(
            args.model_name, 
            quantize=args.quantize, 
            device_map=args.device_map
        )
        
        # 显示GPU使用信息
        display_gpu_info()
        
        # 加载少样本示例
        print(f"Loading few-shot examples from {args.few_shot_dir}")
        few_shot_examples = load_few_shot_examples(args.few_shot_dir, args.ground_truth_file)
        print(f"Loaded {len(few_shot_examples)} few-shot examples")
        
        # 获取对话问题
        conversation = ar_prompts.generate_conversation_questions(include_descriptions=True)
        
        # 处理视频
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} test videos")
        
        for video_file in tqdm(video_files, desc="Processing test videos"):
            if continue_mode and video_file in processed_videos:
                print(f"Skipping already processed video: {video_file}")
                skipped_videos += 1
                continue
                
            try:
                print(f"\nStarting to process: {video_file}")
                video_path = os.path.join(args.video_dir, video_file)
                
                # 加载测试视频帧
                print("Loading test video frames...")
                test_frames, _ = ar_evaluation.load_video_decord(video_path, num_segments=args.num_frames)
                print(f"Loaded {len(test_frames)} frames")
                
                # 如果使用图像卸载，确保帧保持在CPU内存中
                if args.offload_images:
                    # 将所有帧转换为CPU张量
                    cpu_frames = []
                    for frame in test_frames:
                        if hasattr(frame, 'to') and callable(getattr(frame, 'to')):
                            cpu_frames.append(frame.to('cpu'))
                        else:
                            cpu_frames.append(frame)
                    test_frames = cpu_frames
                    print("Images will stay in CPU memory until needed")
                
                # 运行多轮对话
                print(f"Starting conversation with Best-of-{args.bon_samples} strategy...")
                responses, evaluation_results = run_conversation(
                    processor, model, test_frames, few_shot_examples, 
                    conversation, video_file, bon_samples=args.bon_samples,
                    offload_images=args.offload_images
                )
                print("Conversation completed")
                
                # 保存对话日志
                with open(args.output_file, "a") as f:
                    f.write(f"\nProcessing video: {video_path}\n\n")
                    for i, response in enumerate(responses):
                        question = conversation[i] if i < len(conversation) else "Additional question"
                        f.write(f"Question: {question}\n")
                        f.write(f"Answer: {response}\n")
                        f.write("-" * 20 + "\n")
                
                # 保存JSON结果
                ar_evaluation.save_json_results(
                    args.json_output,
                    video_file,
                    evaluation_results,
                    continue_file=args.continue_file if continue_mode else None
                )
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                successful_videos += 1
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory while processing {video_file}. Skipping...")
                with open(args.output_file, "a") as f:
                    f.write(f"\nError processing video {video_path}: CUDA out of memory\n\n")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                failed_videos += 1
                continue
                
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                traceback.print_exc()
                with open(args.output_file, "a") as f:
                    f.write(f"\nError processing video {video_path}: {str(e)}\n\n")
                failed_videos += 1
                continue
    
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        return
    
    finally:
        # 计算并打印总执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 输出执行摘要
        print("\n" + "="*50)
        print("Execution Summary:")
        print(f"Total time: {ar_evaluation.format_time(execution_time)}")
        print(f"Videos: {successful_videos} successful, {failed_videos} failed, {skipped_videos} skipped")
        print("="*50)


if __name__ == "__main__":
    main()