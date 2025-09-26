"""
vLLM Provider for High-Performance Model Serving
Optimized for production deployment with 10x speedup over HuggingFace
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import torch
import numpy as np

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("vLLM not installed. Install with: pip install vllm")

from src.providers.local_model_provider import LocalModelProvider, ModelConfig
from src.providers.base_provider import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig(ModelConfig):
    """Extended configuration for vLLM"""
    # vLLM specific settings
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB of CPU swap space
    enforce_eager: bool = False  # Disable CUDA graphs for debugging
    trust_remote_code: bool = True
    tokenizer_mode: str = "auto"  # auto, slow
    disable_log_requests: bool = False
    quantization: Optional[str] = None  # awq, gptq, squeezellm
    dtype: str = "auto"  # auto, half, float16, bfloat16, float32
    max_model_len: Optional[int] = None
    download_dir: Optional[str] = None
    seed: int = 42

    # Khmer-specific settings
    use_khmer_tokenizer: bool = False
    khmer_tokenizer_path: Optional[str] = None
    handle_mixed_language: bool = True


class VLLMProvider(LocalModelProvider):
    """vLLM provider for high-performance local model serving"""

    def __init__(
        self,
        config: Union[VLLMConfig, Dict[str, Any]],
        **kwargs
    ):
        """
        Initialize vLLM provider

        Args:
            config: vLLM configuration object or dict
            **kwargs: Additional settings
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Run: pip install vllm")

        # Convert dict to VLLMConfig if necessary
        if isinstance(config, dict):
            config = VLLMConfig(**config)

        super().__init__(config, **kwargs)
        self.provider_name = "vllm"
        self.vllm_config = config
        self.llm = None
        self.is_loaded = False

        # Performance metrics
        self.batch_sizes = []
        self.generation_times = []

    def load_model(self) -> None:
        """Load model using vLLM engine"""
        if self.is_loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model from {self.vllm_config.model_path} with vLLM")

        try:
            # Determine tensor parallel size based on available GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            tensor_parallel_size = min(self.vllm_config.tensor_parallel_size, num_gpus) if num_gpus > 0 else 1

            logger.info(f"Using {tensor_parallel_size} GPU(s) for tensor parallelism")

            # Initialize vLLM engine
            self.llm = LLM(
                model=self.vllm_config.model_path,
                tokenizer=self.vllm_config.model_path,
                tokenizer_mode=self.vllm_config.tokenizer_mode,
                trust_remote_code=self.vllm_config.trust_remote_code,
                dtype=self.vllm_config.dtype,
                max_model_len=self.vllm_config.max_model_len,
                gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                swap_space=self.vllm_config.swap_space,
                enforce_eager=self.vllm_config.enforce_eager,
                quantization=self.vllm_config.quantization,
                tensor_parallel_size=tensor_parallel_size,
                seed=self.vllm_config.seed,
                download_dir=self.vllm_config.download_dir,
                disable_log_requests=self.vllm_config.disable_log_requests,
            )

            self.is_loaded = True
            logger.info("vLLM model loaded successfully")

            # Log model info
            model_config = self.llm.llm_engine.model_config
            logger.info(f"Model dtype: {model_config.dtype}")
            logger.info(f"Max model length: {model_config.max_model_len}")
            logger.info(f"Vocab size: {model_config.get_vocab_size()}")

        except Exception as e:
            logger.error(f"Failed to load model with vLLM: {e}")
            raise

    def unload_model(self) -> None:
        """Unload model and free resources"""
        if self.llm:
            logger.info("Unloading vLLM model...")
            del self.llm
            self.llm = None

            # Clean up distributed process groups if using tensor parallelism
            if self.vllm_config.tensor_parallel_size > 1:
                try:
                    destroy_model_parallel()
                except:
                    pass

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            logger.info("Model unloaded successfully")

    async def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """
        Generate response using vLLM

        Args:
            prompt: Input prompt(s)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            stop: Stop sequences
            **kwargs: Additional sampling parameters

        Returns:
            LLMResponse object or list of responses for batch
        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature or self.vllm_config.temperature,
            top_p=top_p or self.vllm_config.top_p,
            top_k=top_k or self.vllm_config.top_k,
            repetition_penalty=repetition_penalty or self.vllm_config.repetition_penalty,
            stop=stop,
            seed=kwargs.get("seed", self.vllm_config.seed),
            use_beam_search=kwargs.get("use_beam_search", False),
            best_of=kwargs.get("best_of", 1),
            n=kwargs.get("n", 1),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
        )

        try:
            # Generate with vLLM
            outputs = self.llm.generate(prompts, sampling_params)

            # Process outputs
            responses = []
            for i, output in enumerate(outputs):
                prompt_text = prompts[i]
                generated_text = output.outputs[0].text

                # Calculate tokens
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)
                total_tokens = prompt_tokens + completion_tokens

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000 / len(outputs)

                response = LLMResponse(
                    text=generated_text,
                    model=self.vllm_config.model_name,
                    provider=self.provider_name,
                    tokens_used=total_tokens,
                    latency_ms=latency_ms,
                    cost=0.0,  # Local inference is free
                    metadata={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "finish_reason": output.outputs[0].finish_reason,
                        "temperature": sampling_params.temperature,
                        "top_p": sampling_params.top_p,
                        "batch_size": len(prompts),
                    }
                )
                responses.append(response)

            # Update statistics
            self.inference_count += len(prompts)
            self.total_inference_time += (time.time() - start_time)
            self.batch_sizes.append(len(prompts))
            self.generation_times.append(time.time() - start_time)

            return responses if is_batch else responses[0]

        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            error_response = LLMResponse(
                text="",
                model=self.vllm_config.model_name,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0.0,
                metadata={},
                error=str(e)
            )
            return [error_response] * len(prompts) if is_batch else error_response

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Stream generation token by token

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Generated tokens
        """
        # Note: vLLM doesn't support true streaming in offline mode
        # For streaming, you would need to use vLLM's API server
        logger.warning("Streaming not supported in offline vLLM mode. Use vLLM API server for streaming.")

        # Generate full response and yield it
        response = self.generate_sync(prompt, max_tokens, temperature, **kwargs)
        if not response.error:
            yield response.text

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model and performance information"""
        base_info = super().get_model_info()

        # Add vLLM specific info
        vllm_info = {
            "engine": "vllm",
            "is_loaded": self.is_loaded,
            "gpu_memory_utilization": self.vllm_config.gpu_memory_utilization,
            "quantization": self.vllm_config.quantization,
            "tensor_parallel_size": self.vllm_config.tensor_parallel_size,
        }

        if self.llm and self.is_loaded:
            model_config = self.llm.llm_engine.model_config
            vllm_info.update({
                "max_model_len": model_config.max_model_len,
                "vocab_size": model_config.get_vocab_size(),
                "model_dtype": str(model_config.dtype),
            })

        # Add performance stats
        if self.batch_sizes:
            vllm_info["performance"] = {
                "avg_batch_size": np.mean(self.batch_sizes),
                "max_batch_size": max(self.batch_sizes),
                "avg_generation_time_s": np.mean(self.generation_times),
                "throughput_tokens_per_second": self._calculate_throughput(),
            }

        base_info.update(vllm_info)
        return base_info

    def _calculate_throughput(self) -> float:
        """Calculate average throughput in tokens per second"""
        if not self.generation_times or self.inference_count == 0:
            return 0.0

        # Estimate tokens generated (rough approximation)
        estimated_tokens = self.inference_count * 100  # Assume 100 tokens average
        total_time = sum(self.generation_times)

        return estimated_tokens / total_time if total_time > 0 else 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> "VLLMProvider":
        """
        Create vLLM provider from a pretrained model

        Args:
            model_name_or_path: HuggingFace model ID or local path
            **kwargs: Additional configuration

        Returns:
            VLLMProvider instance
        """
        config = VLLMConfig(
            model_path=model_name_or_path,
            model_name=model_name_or_path.split("/")[-1],
            **kwargs
        )

        provider = cls(config)
        provider.load_model()
        return provider

    def benchmark_batch_sizes(
        self,
        prompt: str,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        max_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark different batch sizes to find optimal configuration

        Args:
            prompt: Test prompt
            batch_sizes: List of batch sizes to test
            max_tokens: Tokens per generation

        Returns:
            Benchmark results
        """
        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            prompts = [prompt] * batch_size

            start_time = time.time()
            responses = self.generate_sync(prompts, max_tokens=max_tokens)
            end_time = time.time()

            total_time = end_time - start_time
            throughput = batch_size / total_time

            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "total_time_s": total_time,
                "time_per_request_s": total_time / batch_size,
                "throughput_req_per_s": throughput,
            }

            # Check for errors
            if isinstance(responses, list):
                error_count = sum(1 for r in responses if r.error)
                results[f"batch_{batch_size}"]["error_count"] = error_count

        # Find optimal batch size
        best_batch = max(results.items(), key=lambda x: x[1]["throughput_req_per_s"])
        results["optimal_batch_size"] = best_batch[1]["batch_size"]
        results["max_throughput_req_per_s"] = best_batch[1]["throughput_req_per_s"]

        return results