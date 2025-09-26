"""
HuggingFace Provider for GDPval Framework
Free tier LLM provider using HuggingFace Inference API
"""

import os
import logging
import time
import asyncio
from typing import Dict, Optional, Any
import aiohttp
import json

from src.providers.base_provider import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider"""

    # Popular free models on HuggingFace
    FREE_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "falcon-7b": "tiiuae/falcon-7b-instruct",
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",  # Requires approval
        "bloom-3b": "bigscience/bloom-3b",
        "t5-base": "google/flan-t5-base",
        "gpt2": "gpt2-medium"  # Smaller, faster model for testing
    }

    API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{}"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-7b",
        rate_limit: int = 30,
        **kwargs
    ):
        """
        Initialize HuggingFace provider

        Args:
            api_key: HuggingFace API token (optional for public models)
            model: Model name or HF model ID
            rate_limit: Max requests per minute (free tier: ~30)
            **kwargs: Additional configuration
        """
        super().__init__("huggingface", api_key, rate_limit, **kwargs)

        # Get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")

        # Resolve model name
        if model in self.FREE_MODELS:
            self.model_id = self.FREE_MODELS[model]
        else:
            self.model_id = model

        self.api_url = self.API_URL_TEMPLATE.format(self.model_id)

        # Cost estimation (free tier)
        self.cost_per_1k_tokens = 0.0  # Free tier

        logger.info(f"Initialized HuggingFace provider with model: {self.model_id}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response using HuggingFace Inference API

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Prepare payload based on model type
        payload = self._prepare_payload(prompt, max_tokens, temperature, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        # Parse response based on model type
                        generated_text = self._parse_response(response_data)

                        # Estimate tokens
                        input_tokens = self.estimate_tokens(prompt)
                        output_tokens = self.estimate_tokens(generated_text)
                        total_tokens = input_tokens + output_tokens

                        latency_ms = (time.time() - start_time) * 1000

                        return LLMResponse(
                            text=generated_text,
                            model=self.model_id,
                            provider=self.provider_name,
                            tokens_used=total_tokens,
                            latency_ms=latency_ms,
                            cost=self.estimate_cost(input_tokens, output_tokens),
                            metadata={
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "model_type": self._get_model_type()
                            }
                        )

                    elif response.status == 503:
                        # Model is loading
                        estimated_time = response_data.get("estimated_time", 20)
                        error_msg = f"Model is loading. Estimated time: {estimated_time}s"
                        logger.warning(error_msg)

                        # Optionally wait and retry
                        if kwargs.get("wait_for_model", True) and estimated_time < 60:
                            await asyncio.sleep(estimated_time + 5)
                            return await self.generate(prompt, max_tokens, temperature, **kwargs)

                        return LLMResponse(
                            text="",
                            model=self.model_id,
                            provider=self.provider_name,
                            tokens_used=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            cost=0,
                            metadata={},
                            error=error_msg
                        )

                    else:
                        error_msg = f"API error: {response.status} - {response_data}"
                        logger.error(error_msg)

                        return LLMResponse(
                            text="",
                            model=self.model_id,
                            provider=self.provider_name,
                            tokens_used=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            cost=0,
                            metadata={},
                            error=error_msg
                        )

        except asyncio.TimeoutError:
            error_msg = "Request timed out"
            logger.error(error_msg)
            return LLMResponse(
                text="",
                model=self.model_id,
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
                model=self.model_id,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0,
                metadata={},
                error=error_msg
            )

    def _prepare_payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare API payload based on model type

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            API payload dictionary
        """
        model_type = self._get_model_type()

        if model_type in ["text-generation", "text2text-generation"]:
            return {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "do_sample": True,
                    "top_p": kwargs.get("top_p", 0.95),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                    "return_full_text": False
                }
            }
        else:
            # Default payload
            return {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_tokens,
                    "temperature": temperature
                }
            }

    def _parse_response(self, response_data: Any) -> str:
        """
        Parse response based on model output format

        Args:
            response_data: Raw API response

        Returns:
            Generated text
        """
        if isinstance(response_data, list) and len(response_data) > 0:
            if isinstance(response_data[0], dict):
                return response_data[0].get("generated_text", "")
            else:
                return str(response_data[0])
        elif isinstance(response_data, dict):
            return response_data.get("generated_text", "")
        else:
            return str(response_data)

    def _get_model_type(self) -> str:
        """
        Get model type based on model ID

        Returns:
            Model type string
        """
        if "t5" in self.model_id.lower() or "bart" in self.model_id.lower():
            return "text2text-generation"
        elif "gpt" in self.model_id.lower() or "llama" in self.model_id.lower():
            return "text-generation"
        elif "mistral" in self.model_id.lower() or "zephyr" in self.model_id.lower():
            return "text-generation"
        else:
            return "text-generation"

    async def validate_connection(self) -> bool:
        """
        Validate HuggingFace API connection

        Returns:
            True if connection is valid
        """
        try:
            # Try a simple request with minimal input
            response = await self.generate(
                "Hello",
                max_tokens=10,
                wait_for_model=False
            )
            return response.error is None

        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost (free for HuggingFace free tier)

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD (0 for free tier)
        """
        # HuggingFace Inference API is free for most models
        # Rate limits apply instead of costs
        return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "model_type": self._get_model_type(),
            "api_url": self.api_url,
            "rate_limit": self.rate_limiter.max_calls,
            "is_free": True,
            "requires_auth": bool(self.api_key),
            "supported_languages": ["en", "kh"] if "multilingual" in self.model_id.lower() else ["en"],
            "max_context_length": 4096 if "mistral" in self.model_id.lower() else 2048
        }

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """
        List available free models

        Returns:
            Dictionary of model aliases and IDs
        """
        return cls.FREE_MODELS.copy()