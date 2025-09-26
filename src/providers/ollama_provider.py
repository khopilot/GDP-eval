"""
Ollama Provider for GDPval Framework
Local LLM provider using Ollama
"""

import os
import logging
import time
import asyncio
from typing import Dict, Optional, Any, List
import aiohttp
import json

from src.providers.base_provider import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""

    # Popular models available in Ollama
    AVAILABLE_MODELS = {
        "llama2": "llama2:latest",
        "mistral": "mistral:latest",
        "codellama": "codellama:latest",
        "llama2-uncensored": "llama2-uncensored:latest",
        "vicuna": "vicuna:latest",
        "orca-mini": "orca-mini:latest",
        "phi": "phi:latest",
        "tinyllama": "tinyllama:latest",  # Very small, fast model
        "deepseek-coder": "deepseek-coder:latest",
        "gemma": "gemma:latest"
    }

    def __init__(
        self,
        model: str = "llama2",
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ollama provider

        Args:
            model: Model name or tag
            base_url: Ollama API base URL
            **kwargs: Additional configuration
        """
        # No API key needed for local Ollama
        super().__init__("ollama", api_key=None, rate_limit=1000, **kwargs)

        # Get base URL from environment or use default
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

        # Resolve model name
        if model in self.AVAILABLE_MODELS:
            self.model = self.AVAILABLE_MODELS[model]
        else:
            self.model = model

        # API endpoints
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.tags_url = f"{self.base_url}/api/tags"

        # No cost for local execution
        self.cost_per_1k_tokens = 0.0

        logger.info(f"Initialized Ollama provider with model: {self.model}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        use_chat: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using Ollama

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_chat: Use chat endpoint instead of generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        # Choose endpoint
        url = self.chat_url if use_chat else self.generate_url

        # Prepare payload
        if use_chat:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": kwargs.get("top_p", 0.9),
                    "stop": kwargs.get("stop", [])
                }
            }
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": kwargs.get("top_p", 0.9),
                    "stop": kwargs.get("stop", [])
                }
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # Longer timeout for local models
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()

                        # Extract generated text
                        if use_chat:
                            generated_text = response_data.get("message", {}).get("content", "")
                        else:
                            generated_text = response_data.get("response", "")

                        # Get token counts
                        total_tokens = (
                            response_data.get("prompt_eval_count", 0) +
                            response_data.get("eval_count", 0)
                        )

                        # Get generation time
                        total_duration = response_data.get("total_duration", 0) / 1_000_000  # Convert from ns to ms
                        latency_ms = (time.time() - start_time) * 1000

                        return LLMResponse(
                            text=generated_text,
                            model=self.model,
                            provider=self.provider_name,
                            tokens_used=total_tokens,
                            latency_ms=latency_ms,
                            cost=0.0,  # Local execution is free
                            metadata={
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "prompt_eval_count": response_data.get("prompt_eval_count", 0),
                                "eval_count": response_data.get("eval_count", 0),
                                "total_duration_ms": total_duration,
                                "tokens_per_second": response_data.get("eval_count", 0) / (total_duration / 1000) if total_duration > 0 else 0
                            }
                        )

                    else:
                        error_text = await response.text()
                        error_msg = f"Ollama API error: {response.status} - {error_text}"
                        logger.error(error_msg)

                        return LLMResponse(
                            text="",
                            model=self.model,
                            provider=self.provider_name,
                            tokens_used=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            cost=0,
                            metadata={},
                            error=error_msg
                        )

        except aiohttp.ClientConnectorError:
            error_msg = f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0,
                metadata={},
                error=error_msg
            )

        except asyncio.TimeoutError:
            error_msg = "Request timed out - model may be too large or system too slow"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0,
                metadata={},
                error=error_msg
            )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model=self.model,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0,
                metadata={},
                error=error_msg
            )

    async def validate_connection(self) -> bool:
        """
        Validate Ollama connection and model availability

        Returns:
            True if connection is valid and model is available
        """
        try:
            # Check if Ollama is running
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.tags_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        model_names = [m.get("name", "") for m in models]

                        # Check if our model is available
                        if self.model in model_names:
                            logger.info(f"Model {self.model} is available in Ollama")
                            return True
                        else:
                            logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                            return False
                    else:
                        logger.error(f"Failed to get Ollama models: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False

    async def list_available_models(self) -> List[str]:
        """
        List models currently available in Ollama

        Returns:
            List of available model names
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.tags_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        return [m.get("name", "") for m in models]
                    else:
                        logger.error(f"Failed to list models: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful
        """
        try:
            pull_url = f"{self.base_url}/api/pull"
            payload = {"name": model_name}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    pull_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes for large models
                ) as response:
                    if response.status == 200:
                        # Stream the response to show progress
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line)
                                    status = data.get("status", "")
                                    logger.info(f"Pull status: {status}")
                                except json.JSONDecodeError:
                                    pass

                        logger.info(f"Successfully pulled model: {model_name}")
                        return True
                    else:
                        logger.error(f"Failed to pull model: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost (always 0 for local Ollama)

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            0.0 (local execution is free)
        """
        return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self.base_url,
            "is_local": True,
            "is_free": True,
            "requires_auth": False,
            "gpu_required": True,  # Most models benefit from GPU
            "supported_languages": ["en", "kh", "multilingual"],  # Depends on model
            "max_context_length": 4096,  # Varies by model
            "streaming_supported": True
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Ollama service

        Returns:
            Health status information
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Check API status
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])

                        return {
                            "status": "healthy",
                            "service": "ollama",
                            "base_url": self.base_url,
                            "available_models": len(models),
                            "current_model": self.model,
                            "model_loaded": self.model in [m.get("name", "") for m in models]
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "service": "ollama",
                            "base_url": self.base_url,
                            "error": f"API returned status {response.status}"
                        }

        except Exception as e:
            return {
                "status": "offline",
                "service": "ollama",
                "base_url": self.base_url,
                "error": str(e)
            }

    @staticmethod
    def get_installation_instructions() -> str:
        """
        Get Ollama installation instructions

        Returns:
            Installation instructions string
        """
        return """
        To use Ollama provider, you need to install Ollama locally:

        1. Download Ollama from https://ollama.ai
        2. Install for your platform (macOS, Linux, Windows)
        3. Start Ollama service
        4. Pull a model: ollama pull llama2
        5. The service will be available at http://localhost:11434

        Recommended models for GDP evaluation:
        - llama2: General purpose, good for Khmer
        - mistral: Fast and accurate
        - tinyllama: Very fast, lower quality
        - phi: Small but capable
        """