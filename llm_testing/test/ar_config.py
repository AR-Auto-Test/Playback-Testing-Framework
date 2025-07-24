"""
AR Evaluation Configuration

This module provides configuration settings for AR evaluation,
including model-specific parameters and common settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    api_key_env: str
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    
    # Additional model-specific parameters
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ARConfig:
    """Configuration for AR evaluation."""
    # Video processing
    frame_interval: int = 10
    num_frames: int = 8
    
    # Batch processing
    batch_size: int = 1
    max_workers: int = 4
    
    # Best-of-N strategy
    bon_samples: int = 5
    
    # Paths
    temp_frames_dir: str = "temp_frames"
    
    # Models
    models: Dict[str, ModelConfig] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = {}
            
        # Create standard model configurations if not provided
        if not self.models:
            self._create_default_models()
    
    def _create_default_models(self):
        """Create default model configurations."""
        self.models = {
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                api_key_env="OPENAI_API_KEY",
                max_tokens=2048,
                temperature=1.0
            ),
            "o1": ModelConfig(
                name="o1",
                api_key_env="OPENAI_API_KEY",
                max_tokens=2048,
                temperature=1.0
            ),
            "gemini-2-pro": ModelConfig(
                name="gemini-2.0-pro-exp-02-05",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                temperature=1.0,
                top_p=0.95,
                top_k=40
            ),
            "gemini-2-flash-thinking": ModelConfig(
                name="gemini-2.0-flash-thinking-exp-01-21",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                temperature=1.0,
                top_p=0.95,
                top_k=40
            ),
            "claude-3-7-sonnet": ModelConfig(
                name="claude-3-7-sonnet-20250219",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=20000,
                temperature=1.0
            )
        }
    
    def get_model_config(self, model_key: str) -> ModelConfig:
        """Get configuration for a specific model.
        
        Args:
            model_key: Key of the model in the models dictionary
            
        Returns:
            ModelConfig for the specified model
        """
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found in configuration")
        return self.models[model_key]
    
    def get_api_key(self, model_key: str) -> str:
        """Get API key for a specific model from environment variables.
        
        Args:
            model_key: Key of the model in the models dictionary
            
        Returns:
            API key for the specified model
        """
        model_config = self.get_model_config(model_key)
        api_key = os.environ.get(model_config.api_key_env)
        if not api_key:
            raise ValueError(f"API key environment variable '{model_config.api_key_env}' not set")
        return api_key


# Default configuration instance
DEFAULT_CONFIG = ARConfig()