"""
Base Provider for LLM APIs
Abstract base class for all LLM providers
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standard response format for all LLM providers"""
    text: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    cost: float
    metadata: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_calls_per_minute: int):
        """
        Initialize rate limiter

        Args:
            max_calls_per_minute: Maximum API calls allowed per minute
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]

            if len(self.calls) >= self.max_calls:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = 60 - (now - oldest_call) + 0.1
                logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                # Retry after waiting
                return await self.acquire()

            self.calls.append(now)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        rate_limit: int = 60,
        **kwargs
    ):
        """
        Initialize base LLM provider

        Args:
            provider_name: Name of the provider
            api_key: API key if required
            rate_limit: Max requests per minute
            **kwargs: Additional provider-specific configuration
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit)
        self.config = kwargs
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.error_count = 0

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from the LLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that the provider is accessible

        Returns:
            True if connection is valid
        """
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a request

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pass

    async def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with automatic retry on failure

        Args:
            prompt: Input prompt
            max_retries: Maximum number of retries
            **kwargs: Generation parameters

        Returns:
            LLMResponse object
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                await self.rate_limiter.acquire()

                # Generate response
                response = await self.generate(prompt, **kwargs)

                if response.error:
                    last_error = response.error
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {response.error}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # Success - update statistics
                    self.total_requests += 1
                    self.total_tokens += response.tokens_used
                    self.total_cost += response.cost
                    return response

            except Exception as e:
                last_error = str(e)
                self.error_count += 1
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} raised exception: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

        # All retries failed
        self.error_count += 1
        return LLMResponse(
            text="",
            model=self.config.get('model', 'unknown'),
            provider=self.provider_name,
            tokens_used=0,
            latency_ms=0,
            cost=0,
            metadata={},
            error=f"Failed after {max_retries} attempts. Last error: {last_error}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get provider statistics

        Returns:
            Dictionary with usage statistics
        """
        return {
            "provider": self.provider_name,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "average_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
            "average_cost_per_request": self.total_cost / max(self.total_requests, 1)
        }

    def reset_statistics(self):
        """Reset usage statistics"""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.error_count = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token per 4 characters
        # This varies by model and language
        return len(text) // 4