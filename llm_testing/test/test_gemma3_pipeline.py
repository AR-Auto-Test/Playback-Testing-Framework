"""
# 标准执行（从头开始，不继续之前的任务）
python test_gemma3_pipeline.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "google/gemma-3-27b-it" \
  --output_file responses/output.txt \
  --json_output responses/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --use_flash_attn False \
  --use_fewshot True

# 继续执行（继续处理之前未完成的视频）
python test_gemma3_pipeline.py \
  --video_dir data/exp/video_clips_6s_3 \
  --few_shot_dir data/exp/few_shot_examples_img \
  --ground_truth_file data/exp/few_shot_img.json \
  --model_name "google/gemma-3-27b-it" \
  --output_file results/output.txt \
  --json_output results/evaluation.json \
  --num_frames 8 \
  --bon_samples 5 \
  --continue_file results/evaluation_20240508_143022.json \
  --use_flash_attn False \
  --use_fewshot False
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
from transformers import pipeline

# 导入自定义模块
import ar_prompts
import ar_evaluation

# 设置常量
BATCH_SIZE = 1  # 视频批处理大小

class FewShotExample:
    """用于存储和处理少样本示例的类"""
    def __init__(self, image_path, ground_truth):
        self.image_path = image_path
        self.ground_truth = ground_truth
        self.processed_image = None

    def load_image(self):
        """加载图像并存储为PIL格式"""
        self.processed_image = Image.open(self.image_path).convert('RGB')
        return self.processed_image


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test Gemma-3 model with image-based few-shot learning using pipeline")
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
    parser.add_argument("--use_flash_attn", type=bool, default=False,
                       help="Whether to use flash attention (disable for multi-GPU)")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map for model distribution")
    parser.add_argument("--use_fewshot", type=bool, default=True,
                       help="Whether to use few-shot learning examples")
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
            example.load_image()
            examples.append(example)
        else:
            print(f"Warning: No ground truth found for {image_file}")
    
    return examples


def run_conversation_with_pipeline(pipe, test_frames, few_shot_examples, conversation, video_file, bon_samples=5, use_fewshot=True):
    """使用pipeline运行与Gemma3模型的多轮对话"""
    # 主动清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    responses = []
    evaluation_results = {}
    
    # 创建系统消息
    system_prompt = ar_prompts.generate_system_prompt()
    few_shot_prompt = ar_prompts.generate_few_shot_prompt(few_shot_examples) if use_fewshot else ""
    full_system_prompt = system_prompt + "\n\n" + few_shot_prompt
    
    # 初始化对话历史
    history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": full_system_prompt}]
        }
    ]
    
    # 初始验证问题(前两个问题)
    for turn in range(2):
        # 创建包含测试视频帧的消息
        message_content = [{"type": "text", "text": conversation[turn]}]
        
        # 仅为第一个问题添加图像
        if turn == 0:
            # 首先添加少样本示例图像（如果启用）
            if use_fewshot:
                for example in few_shot_examples:
                    message_content.insert(0, {"type": "image", "image": example.processed_image})
                
            # 然后添加测试帧
            for frame in test_frames:
                message_content.append({"type": "image", "image": frame})
        
        # 将消息添加到历史记录
        history.append(
            {
                "role": "user", 
                "content": message_content
            }
        )
        
        try:
            # 使用pipeline生成响应
            output = pipe(text=history, max_new_tokens=1024)
            
            # 提取响应文本
            response = output[0]["generated_text"][-1]["content"]
            responses.append(response)
            
            # 添加助手响应到历史记录
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })
            
            # 每次对话完成后清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error in conversation turn {turn}: {str(e)}")
            # 如果出错，添加一个空响应以保持对话流程
            responses.append("I'm unable to respond to this question at the moment.")
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "I'm unable to respond to this question at the moment."}]
            })
    
    # 保存验证问题后的历史记录
    initial_history = history.copy()
    
    # 六个评估问题（使用Best-of-N策略）
    evaluation_responses = []
    metrics = ar_prompts.METRICS
    
    for i in range(6):  # 六个评估问题
        # 主动清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 重置历史记录到验证问题后
        current_history = initial_history.copy()
        
        # 获取带有指标描述的具体问题
        question_idx = i + 2  # 偏移2用于验证问题
        question_with_metrics = conversation[question_idx]
        
        # 使用不同参数生成N个样本用于Best-of-N
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
                params = ar_evaluation.get_diverse_params(n, bon_samples)
                
                # 使用pipeline生成响应
                output = pipe(
                    text=bon_history, 
                    max_new_tokens=params["max_new_tokens"],
                    do_sample=params["do_sample"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"]
                )
                
                # 提取响应文本
                response = output[0]["generated_text"][-1]["content"]
                bon_responses.append(response)
                
                # 尝试提取JSON
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
                
                # 每生成一个样本后立即清理内存
                # 这样不会影响后续操作，因为每个样本都是独立生成的
                del output
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Memory freed after BoN sample {n+1}/{bon_samples}")
                
            except Exception as e:
                print(f"Error in BoN sample {n} for metric {metrics[i]}: {str(e)}")
                # 如果这个样本失败，继续尝试下一个
                # 确保在异常情况下也进行内存清理
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # 使用多数投票选择最佳响应
        best_response, best_json = ar_evaluation.select_best_response(bon_responses, bon_json_data)
        evaluation_responses.append(best_response)
        
        # 存储结果
        if best_json:
            evaluation_results[metrics[i]] = best_json
        else:
            # 如果没有找到有效的JSON，创建回退结果
            evaluation_results[metrics[i]] = {
                "Video_name": video_file,
                "Metrics": metrics[i],
                "Issue": False,  # 默认值
                "Reason": "Could not determine from model output",
                "error": "No valid JSON found"
            }
    
        # 更新历史以用于下一个指标评估
        current_history.append({
            "role": "user",
            "content": [{"type": "text", "text": question_with_metrics}]
        })
        current_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": best_response}]
        })
        
        # 每完成一个指标后进行一次额外的内存清理
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Completed evaluation for metric: {metrics[i]}")
    
    # 最后一个关于其他问题的问题（使用完整历史记录）
    try:
        # 主动清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        final_history = current_history.copy()
        
        # 添加最后一个问题
        final_history.append({
            "role": "user",
            "content": [{"type": "text", "text": conversation[8]}]
        })
        
        # 生成最后的响应
        output = pipe(text=final_history, max_new_tokens=1024)
        final_response = output[0]["generated_text"][-1]["content"]
    except Exception as e:
        print(f"Error in final question: {str(e)}")
        final_response = "I'm unable to respond to this question at the moment."
    
    # 收集所有响应
    responses.extend(evaluation_responses)
    responses.append(final_response)
    
    return responses, evaluation_results

def main():
    # 记录程序开始时间
    start_time = time.time()
    
    args = parse_args()
    
    # 生成带时间戳的输出文件名
    args.output_file = ar_evaluation.generate_timestamp_filename(args.output_file)
    args.json_output = ar_evaluation.generate_timestamp_filename(args.json_output)
    
    print(f"Using output file: {args.output_file}")
    print(f"Using JSON output file: {args.json_output}")
    print(f"Using few-shot learning: {args.use_fewshot}")
    
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
    
    try:
        # 初始化pipeline
        print(f"Loading Gemma3 model via pipeline: {args.model_name}")
        pipe = pipeline(
            "image-text-to-text",
            model=args.model_name,
            model_kwargs={
                "device_map": args.device_map,
                "attn_implementation": "sdpa" if not args.use_flash_attn else "flash_attention_2"
            },
            torch_dtype=torch.bfloat16
        )
        
        # 加载少样本示例（仅在启用few-shot的情况下）
        few_shot_examples = []
        if args.use_fewshot:
            print(f"Loading few-shot examples from {args.few_shot_dir}")
            few_shot_examples = load_few_shot_examples(args.few_shot_dir, args.ground_truth_file)
            print(f"Loaded {len(few_shot_examples)} few-shot examples")
        else:
            print("Few-shot learning disabled, skipping example loading")
        
        # 获取对话问题，包含详细的指标描述
        conversation = ar_prompts.generate_conversation_questions(include_descriptions=True)
        
        # 处理视频
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        print(f"Found {len(video_files)} test videos")
        
        for video_file in tqdm(video_files, desc="Processing test videos"):
            if continue_mode and video_file in processed_videos:
                print(f"Skipping already processed video: {video_file}")
                continue
                
            try:
                print(f"\nStarting to process: {video_file}")
                video_path = os.path.join(args.video_dir, video_file)
                
                # 加载测试视频帧
                print("Loading test video frames...")
                test_frames, _ = ar_evaluation.load_video_decord(video_path, num_segments=args.num_frames)
                print(f"Loaded {len(test_frames)} frames")
                
                # 运行多轮对话
                print(f"Starting conversation with Best-of-{args.bon_samples} strategy...")
                responses, evaluation_results = run_conversation_with_pipeline(
                    pipe, test_frames, few_shot_examples, 
                    conversation, video_file, 
                    bon_samples=args.bon_samples,
                    use_fewshot=args.use_fewshot
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
                
                # 清理CUDA缓存以释放内存
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
    
    finally:
        # 计算并打印总执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {ar_evaluation.format_time(execution_time)} ({execution_time:.2f} seconds)")

if __name__ == "__main__":
    main()