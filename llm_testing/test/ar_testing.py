"""
AR Testing Base Class

This module provides a base class for AR testing with different models.
"""

import os
import json
import time
import argparse
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm

from ar_config import ARConfig, DEFAULT_CONFIG
from ar_evaluation import (
    read_video_frames, save_json_results, extract_json_content, 
    get_diverse_params, select_best_response, generate_timestamp_filename,
    get_processed_videos
)
from ar_prompts import (
    generate_system_prompt, generate_conversation_questions,
    METRICS
)


class ARTester(ABC):
    """Base class for AR testing with different models."""
    
    def __init__(self, config: ARConfig = None):
        """Initialize ARTester.
        
        Args:
            config: Configuration object, uses DEFAULT_CONFIG if None
        """
        self.config = config or DEFAULT_CONFIG
        self.file_lock = Lock()  # For thread-safe file operations
    
    @abstractmethod
    def initialize_client(self):
        """Initialize the client for the specific model."""
        pass
    
    @abstractmethod
    def process_video(self, video_path: str, output_file: str, json_output_file: str) -> None:
        """Process a single video with the model.
        
        Args:
            video_path: Path to the video file
            output_file: Path to save the conversation log
            json_output_file: Path to save the JSON results
        """
        pass
    
    def process_videos(self, args: argparse.Namespace) -> None:
        """Process multiple videos with the model.
        
        Args:
            args: Command-line arguments
        """
        # Initialize client
        self.initialize_client()
        
        # Get video paths
        video_files = [
            f for f in os.listdir(args.video_dir)
            if f.endswith((".mp4", ".mov"))
        ]
        
        # Process videos in continue mode if specified
        processed_videos = set()
        if args.mode == "continue":
            # Parse log file to track processed videos
            if os.path.exists(args.output_file):
                with open(args.output_file, 'r') as f:
                    for line in f:
                        if line.startswith("Processing video:"):
                            video_name = line.split("/")[-1].strip()
                            processed_videos.add(video_name)
            
            # Also check the JSON output file
            processed_videos.update(get_processed_videos(args.json_output_file))
            
            # Filter out already processed videos
            video_files = [f for f in video_files if f not in processed_videos]
        
        # Initialize output files in discard mode
        if args.mode == "discard":
            with open(args.output_file, 'w') as f:
                pass
            with open(args.json_output_file, 'w') as f:
                pass
        
        if not video_files:
            print("No videos to process.")
            return
        
        print(f"Found {len(processed_videos)} videos already processed.")
        print(f"Total videos to process: {len(video_files)}")
        
        # Process videos with thread pool
        video_paths = [os.path.join(args.video_dir, f) for f in video_files]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_video, 
                    video_path, 
                    args.output_file,
                    args.json_output_file
                ): video_path 
                for video_path in video_paths
            }
            
            for i, future in enumerate(as_completed(futures), start=1):
                video_path = futures[future]
                try:
                    future.result()
                    print(f"Finished processing {video_path} ({i}/{len(video_paths)})")
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    
    @staticmethod
    def add_base_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add base arguments to the argument parser.
        
        Args:
            parser: Argument parser to add arguments to
            
        Returns:
            Updated argument parser
        """
        parser.add_argument(
            "--video_dir", 
            type=str, 
            default="../video_clips_6s_3",
            help="Directory containing video files"
        )
        parser.add_argument(
            "--output_file", 
            type=str, 
            default="output.txt",
            help="File to save conversation logs"
        )
        parser.add_argument(
            "--json_output_file", 
            type=str, 
            default="evaluation.json",
            help="File to save evaluation results"
        )
        parser.add_argument(
            "--mode", 
            choices=["discard", "continue"], 
            default="discard",
            help="Choose the working mode: 'discard' to start fresh, 'continue' to skip processed videos"
        )
        parser.add_argument(
            "--bon_samples", 
            type=int, 
            default=5,
            help="Number of samples to generate for Best-of-N strategy"
        )
        parser.add_argument(
            "--add_timestamp", 
            action="store_true",
            help="Add timestamp to output filenames"
        )
        
        return parser


class BestOfNMixin:
    """Mixin for Best-of-N strategy implementation."""
    
    def apply_best_of_n(self, 
                        generate_fn, 
                        question: str, 
                        context: Any,
                        bon_samples: int, 
                        video_name: str,
                        metric: str) -> Tuple[str, Dict]:
        """Apply Best-of-N strategy to generate multiple responses and select the best one.
        
        Args:
            generate_fn: Function to generate a response
            question: Question to ask
            context: Context for the question (e.g., history, images)
            bon_samples: Number of samples to generate
            video_name: Name of the video
            metric: Metric being evaluated
            
        Returns:
            Tuple of (best response text, best JSON data)
        """
        bon_responses = []
        bon_json_data = []
        
        # Generate multiple responses
        for n in range(bon_samples):
            # Get diverse parameters for this sample
            params = get_diverse_params(n, bon_samples)
            
            # Generate response
            try:
                response = generate_fn(question, context, params)
                bon_responses.append(response)
                
                # Try to extract JSON from response
                json_content = extract_json_content(response)
                if json_content:
                    json_data = json.loads(json_content)
                    json_data['Video_name'] = video_name
                    if 'Metrics' not in json_data:
                        json_data['Metrics'] = metric
                    bon_json_data.append(json_data)
            except Exception as e:
                print(f"Error generating sample {n} for {metric}: {e}")
        
        # Select best response using majority voting
        return select_best_response(bon_responses, bon_json_data)