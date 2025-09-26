"""
Grok (xAI) Provider for GDP Evaluation
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from src.providers.base_provider import BaseLLMProvider as BaseProvider, LLMResponse

logger = logging.getLogger(__name__)


class GrokProvider(BaseProvider):
    """Provider for Grok models via xAI API"""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-3",  # Updated to latest model
        base_url: str = "https://api.x.ai/v1",
        timeout: int = 120  # Increased for reliable completion
    ):
        """
        Initialize Grok provider

        Args:
            api_key: xAI API key
            model: Model name (grok-3, grok-4-latest)
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        super().__init__(provider_name="grok", model_name=model)
        self.api_key = api_key
        self.model = model
        self.model_name = model
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from Grok

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        # Add any additional parameters
        for key in ["top_p", "top_k", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Grok API error: {response.status} - {error_text}")
                        return LLMResponse(
                            text="",
                            model=self.model_name,
                            provider=self.provider_name,
                            tokens_used=0,
                            latency_ms=(time.time() - start_time) * 1000,
                            cost=0.0,
                            metadata={"status": response.status, "error": error_text, "success": False},
                            error=f"API error: {response.status}"
                        )

                    data = await response.json()

                    # Extract response
                    generated_text = data["choices"][0]["message"]["content"]

                    # Get token usage
                    input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                    output_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    total_tokens = data.get("usage", {}).get("total_tokens", 0)

                    # Calculate cost
                    cost = self.estimate_cost(input_tokens, output_tokens)

                    return LLMResponse(
                        text=generated_text,
                        model=self.model_name,
                        provider=self.provider_name,
                        tokens_used=total_tokens,
                        latency_ms=(time.time() - start_time) * 1000,
                        cost=cost,
                        metadata={
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "model": data.get("model"),
                            "usage": data.get("usage", {}),
                            "success": True
                        }
                    )

        except asyncio.TimeoutError:
            logger.error(f"Grok request timed out after {self.timeout}s")
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=self.timeout * 1000,
                cost=0.0,
                metadata={"error_type": "timeout", "success": False},
                error="Request timeout"
            )
        except Exception as e:
            logger.error(f"Grok generation error: {e}")
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0.0,
                metadata={"error_type": "exception", "exception": str(e), "success": False},
                error=str(e)
            )

    async def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts

        Args:
            prompts: List of prompts
            **kwargs: Generation parameters

        Returns:
            List of LLM responses
        """
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "base_url": self.base_url,
            "capabilities": {
                "streaming": True,
                "function_calling": True,
                "vision": False,
                "max_context": 100000,  # Grok has 100k context
                "languages": ["en", "km", "multilingual"]
            }
        }

    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = await self.generate(
                "Hi",
                max_tokens=10,
                temperature=0
            )
            # Check if response has text (successful) or error
            return bool(response.text)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for token usage

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Grok pricing (estimated - adjust based on actual pricing)
        input_price_per_1k = 0.001  # $0.001 per 1K input tokens
        output_price_per_1k = 0.002  # $0.002 per 1K output tokens

        cost = (input_tokens * input_price_per_1k / 1000) + \
               (output_tokens * output_price_per_1k / 1000)
        return cost

    async def validate_connection(self) -> bool:
        """
        Validate provider connection

        Returns:
            True if connection is valid
        """
        return await self.test_connection()