"""
Base Provider for Local Model Serving
Foundation for vLLM, TGI, and other local serving solutions
"""

import os
import logging
import psutil
import GPUtil
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import json

from src.providers.base_provider import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for local model serving"""
    model_path: str
    model_name: str
    device: str = "auto"  # auto, cuda, cuda:0, cpu
    dtype: str = "auto"  # auto, float16, bfloat16, float32
    max_memory: Optional[Dict[str, str]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    use_flash_attention: bool = True
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_batch_size: int = 256
    max_input_length: int = 2048
    max_total_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalModelProvider(BaseLLMProvider):
    """Base class for local model serving providers"""

    def __init__(
        self,
        config: ModelConfig,
        **kwargs
    ):
        """
        Initialize local model provider

        Args:
            config: Model configuration
            **kwargs: Additional provider-specific settings
        """
        super().__init__(
            provider_name="local_model",
            api_key=None,  # No API key needed for local models
            rate_limit=1000,  # High rate limit for local serving
            **kwargs
        )

        self.model_config = config
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device(config.device)
        self.dtype = self._setup_dtype(config.dtype)

        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0
        self.peak_memory_usage = 0

    def _setup_device(self, device_str: str) -> torch.device:
        """
        Set up compute device

        Args:
            device_str: Device specification string

        Returns:
            torch.device object
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                # Select GPU with most free memory
                gpus = GPUtil.getGPUs()
                if gpus:
                    best_gpu = max(gpus, key=lambda x: x.memoryFree)
                    device = torch.device(f"cuda:{best_gpu.id}")
                    logger.info(f"Using GPU {best_gpu.id} with {best_gpu.memoryFree:.1f}MB free memory")
                else:
                    device = torch.device("cuda")
                    logger.info("Using default CUDA device")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        else:
            device = torch.device(device_str)
            logger.info(f"Using specified device: {device}")

        return device

    def _setup_dtype(self, dtype_str: str) -> torch.dtype:
        """
        Set up tensor dtype

        Args:
            dtype_str: Data type specification string

        Returns:
            torch.dtype object
        """
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "int8": torch.int8,
        }

        if dtype_str == "auto":
            if torch.cuda.is_available():
                # Use float16 for GPU by default
                dtype = torch.float16
                logger.info("Using float16 for GPU inference")
            else:
                # Use float32 for CPU
                dtype = torch.float32
                logger.info("Using float32 for CPU inference")
        else:
            dtype = dtype_map.get(dtype_str, torch.float32)
            logger.info(f"Using specified dtype: {dtype}")

        return dtype

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory (implemented by subclasses)"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory to free resources"""
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics

        Returns:
            Dictionary with memory usage information
        """
        memory_stats = {
            "cpu_ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "cpu_ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_ram_percent": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_stats[f"gpu_{i}_used_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                memory_stats[f"gpu_{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024**3)
                memory_stats[f"gpu_{i}_total_gb"] = torch.cuda.get_device_properties(i).total_memory / (1024**3)

            # Get GPU utilization
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                memory_stats[f"gpu_{gpu.id}_utilization_percent"] = gpu.load * 100
                memory_stats[f"gpu_{gpu.id}_memory_percent"] = gpu.memoryUtil * 100

        return memory_stats

    def warmup(self, num_iterations: int = 3) -> None:
        """
        Warm up the model with dummy inputs

        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model with {num_iterations} iterations...")

        dummy_prompts = [
            "Hello, how are you?",
            "តើអ្នកសុខសប្បាយទេ?",  # Khmer: How are you?
            "What is the capital of Cambodia?",
        ]

        for i in range(num_iterations):
            for prompt in dummy_prompts:
                try:
                    # Run inference without counting it in statistics
                    _ = self.generate_sync(prompt, max_tokens=10)
                except Exception as e:
                    logger.warning(f"Warmup iteration {i} failed: {e}")

        logger.info("Model warmup completed")

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Synchronous generation (wrapper for async generate)

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate(prompt, max_tokens, temperature, **kwargs)
            )
        finally:
            loop.close()

    async def validate_connection(self) -> bool:
        """
        Validate that the model is loaded and ready

        Returns:
            True if model is ready
        """
        if self.model is None:
            logger.warning("Model not loaded")
            return False

        # Check if model responds to simple query
        try:
            response = await self.generate("Hello", max_tokens=5)
            return response.error is None
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost (always 0 for local models)

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            0.0 (local serving is free)
        """
        return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information

        Returns:
            Dictionary with model details
        """
        memory_stats = self.get_memory_usage()

        return {
            "provider": self.provider_name,
            "model_path": self.model_config.model_path,
            "model_name": self.model_config.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "max_batch_size": self.model_config.max_batch_size,
            "max_input_length": self.model_config.max_input_length,
            "tensor_parallel_size": self.model_config.tensor_parallel_size,
            "use_flash_attention": self.model_config.use_flash_attention,
            "is_loaded": self.model is not None,
            "inference_count": self.inference_count,
            "avg_inference_time_ms": (self.total_inference_time / max(self.inference_count, 1)) * 1000,
            "memory_usage": memory_stats,
            "metadata": self.model_config.metadata,
        }

    def save_config(self, filepath: str) -> None:
        """
        Save model configuration to file

        Args:
            filepath: Path to save configuration
        """
        config_dict = {
            "model_path": self.model_config.model_path,
            "model_name": self.model_config.model_name,
            "device": self.model_config.device,
            "dtype": self.model_config.dtype,
            "max_batch_size": self.model_config.max_batch_size,
            "max_input_length": self.model_config.max_input_length,
            "max_total_tokens": self.model_config.max_total_tokens,
            "tensor_parallel_size": self.model_config.tensor_parallel_size,
            "use_flash_attention": self.model_config.use_flash_attention,
            "metadata": self.model_config.metadata,
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> ModelConfig:
        """
        Load model configuration from file

        Args:
            filepath: Path to configuration file

        Returns:
            ModelConfig object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return ModelConfig(**config_dict)

    def benchmark(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark model performance

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens per generation
            num_iterations: Number of iterations

        Returns:
            Benchmark results
        """
        import time
        import numpy as np

        logger.info(f"Starting benchmark with {len(prompts)} prompts, {num_iterations} iterations")

        latencies = []
        tokens_per_second = []
        memory_usage = []

        for i in range(num_iterations):
            for prompt in prompts:
                start_time = time.time()
                start_memory = self.get_memory_usage()

                response = self.generate_sync(prompt, max_tokens=max_tokens)

                end_time = time.time()
                end_memory = self.get_memory_usage()

                if not response.error:
                    latency = (end_time - start_time) * 1000  # ms
                    latencies.append(latency)

                    # Estimate tokens generated
                    output_tokens = len(response.text.split())
                    tps = output_tokens / (latency / 1000)
                    tokens_per_second.append(tps)

                    # Track memory delta
                    if "gpu_0_used_gb" in start_memory and "gpu_0_used_gb" in end_memory:
                        mem_delta = end_memory["gpu_0_used_gb"] - start_memory["gpu_0_used_gb"]
                        memory_usage.append(mem_delta)

        results = {
            "num_prompts": len(prompts),
            "num_iterations": num_iterations,
            "total_requests": len(prompts) * num_iterations,
            "latency_ms": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
            },
            "throughput": {
                "tokens_per_second_mean": np.mean(tokens_per_second),
                "tokens_per_second_median": np.median(tokens_per_second),
            },
            "memory": {
                "delta_gb_mean": np.mean(memory_usage) if memory_usage else 0,
                "current_usage": self.get_memory_usage(),
            }
        }

        logger.info(f"Benchmark complete: {results['latency_ms']['mean']:.2f}ms mean latency")
        return results