"""
Enterprise-Grade Grok Provider with Advanced Features
Production-ready implementation for professional GDP evaluation
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

from src.providers.base_provider import BaseLLMProvider as BaseProvider, LLMResponse

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for timeout calculation"""
    SIMPLE = 1
    BASIC = 2
    MODERATE = 3
    COMPLEX = 4
    EXPERT = 5


class EnterpriseGrokProvider(BaseProvider):
    """Enterprise Grok provider with professional features"""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-3",
        base_url: str = "https://api.x.ai/v1",
        enable_retry: bool = True,
        max_retries: int = 3,
        enable_caching: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize enterprise Grok provider

        Args:
            api_key: xAI API key
            model: Model name
            base_url: API base URL
            enable_retry: Enable retry logic
            max_retries: Maximum retry attempts
            enable_caching: Enable response caching
            enable_monitoring: Enable performance monitoring
        """
        super().__init__(provider_name="grok_enterprise", model_name=model)
        self.api_key = api_key
        self.model = model
        self.model_name = model
        self.base_url = base_url
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self.enable_monitoring = enable_monitoring

        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_latency": 0.0
        }

        # Response cache
        self.cache = {} if enable_caching else None

        # Connection pool session
        self.session = None

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def calculate_adaptive_timeout(self, complexity: int, is_bilingual: bool = False) -> int:
        """
        Calculate timeout based on task complexity

        Args:
            complexity: Task complexity level (1-5)
            is_bilingual: Whether task requires bilingual processing

        Returns:
            Timeout in seconds
        """
        base_timeouts = {
            1: 30,   # Simple
            2: 60,   # Basic
            3: 120,  # Moderate
            4: 180,  # Complex
            5: 240   # Expert
        }

        timeout = base_timeouts.get(complexity, 120)

        # Add extra time for bilingual tasks
        if is_bilingual:
            timeout = int(timeout * 1.5)

        logger.info(f"Adaptive timeout: {timeout}s for complexity {complexity}, bilingual: {is_bilingual}")
        return timeout

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create connection pool session"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    async def generate_with_retry(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        complexity: int = 3,
        is_bilingual: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate with retry logic and adaptive timeout

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            complexity: Task complexity
            is_bilingual: Bilingual task flag
            system_prompt: System prompt
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        # Check cache first
        cache_key = f"{prompt[:100]}_{max_tokens}_{temperature}"
        if self.cache and cache_key in self.cache:
            logger.info("Cache hit - returning cached response")
            cached = self.cache[cache_key]
            cached.metadata["cache_hit"] = True
            return cached

        timeout = self.calculate_adaptive_timeout(complexity, is_bilingual)
        last_error = None

        for attempt in range(self.max_retries if self.enable_retry else 1):
            if attempt > 0:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retry attempt {attempt + 1} after {wait_time}s")
                await asyncio.sleep(wait_time)

            try:
                response = await self._generate_internal(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    system_prompt=system_prompt,
                    **kwargs
                )

                if response.text:  # Successful response
                    # Update metrics
                    if self.enable_monitoring:
                        self._update_metrics(response, success=True)

                    # Cache successful response
                    if self.cache is not None:
                        self.cache[cache_key] = response

                    return response

                last_error = response.error

            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt + 1} failed: {e}")

        # All retries failed
        failed_response = LLMResponse(
            text="",
            model=self.model_name,
            provider=self.provider_name,
            tokens_used=0,
            latency_ms=timeout * 1000,
            cost=0.0,
            metadata={"retries": self.max_retries, "final_error": last_error},
            error=f"Failed after {self.max_retries} retries: {last_error}"
        )

        if self.enable_monitoring:
            self._update_metrics(failed_response, success=False)

        return failed_response

    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Internal generation method"""
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

        # Add optional parameters
        for key in ["top_p", "top_k", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        start_time = time.time()
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return LLMResponse(
                        text="",
                        model=self.model_name,
                        provider=self.provider_name,
                        tokens_used=0,
                        latency_ms=latency_ms,
                        cost=0.0,
                        metadata={"status": response.status, "error": error_text},
                        error=f"API error {response.status}"
                    )

                data = await response.json()

                # Extract response
                generated_text = data["choices"][0]["message"]["content"]

                # Token usage
                input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                output_tokens = data.get("usage", {}).get("completion_tokens", 0)
                total_tokens = data.get("usage", {}).get("total_tokens", 0)

                # Calculate cost with enterprise pricing
                cost = self.calculate_enterprise_cost(input_tokens, output_tokens)

                return LLMResponse(
                    text=generated_text,
                    model=self.model_name,
                    provider=self.provider_name,
                    tokens_used=total_tokens,
                    latency_ms=latency_ms,
                    cost=cost,
                    metadata={
                        "finish_reason": data["choices"][0].get("finish_reason"),
                        "model": data.get("model"),
                        "usage": data.get("usage", {}),
                        "success": True,
                        "timeout_used": timeout
                    }
                )

        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout}s")
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=timeout * 1000,
                cost=0.0,
                metadata={"error_type": "timeout", "timeout": timeout},
                error=f"Timeout after {timeout}s"
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider_name,
                tokens_used=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost=0.0,
                metadata={"error_type": "exception", "exception": str(e)},
                error=str(e)
            )

    def calculate_enterprise_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost with enterprise pricing tiers

        Args:
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in USD
        """
        # Enterprise pricing (volume discounts)
        if self.metrics["total_tokens"] > 1000000:  # 1M+ tokens
            input_price = 0.0008  # $0.80 per 1M
            output_price = 0.0016  # $1.60 per 1M
        elif self.metrics["total_tokens"] > 100000:  # 100K+ tokens
            input_price = 0.0009  # $0.90 per 1M
            output_price = 0.0018  # $1.80 per 1M
        else:  # Standard pricing
            input_price = 0.001  # $1.00 per 1M
            output_price = 0.002  # $2.00 per 1M

        cost = (input_tokens * input_price + output_tokens * output_price) / 1000
        return cost

    def _update_metrics(self, response: LLMResponse, success: bool):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1

        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        self.metrics["total_tokens"] += response.tokens_used
        self.metrics["total_cost"] += response.cost

        # Update average latency
        n = self.metrics["total_requests"]
        current_avg = self.metrics["average_latency"]
        self.metrics["average_latency"] = (
            (current_avg * (n - 1) + response.latency_ms) / n
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        success_rate = (
            self.metrics["successful_requests"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0
        )

        return {
            **self.metrics,
            "success_rate": success_rate,
            "cache_size": len(self.cache) if self.cache else 0
        }

    async def batch_generate_optimized(
        self,
        prompts: List[str],
        complexities: Optional[List[int]] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Optimized batch generation with complexity awareness

        Args:
            prompts: List of prompts
            complexities: List of complexity levels
            **kwargs: Generation parameters

        Returns:
            List of responses
        """
        if complexities is None:
            complexities = [3] * len(prompts)  # Default moderate complexity

        tasks = []
        for prompt, complexity in zip(prompts, complexities):
            task = self.generate_with_retry(
                prompt=prompt,
                complexity=complexity,
                **kwargs
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on API connection"""
        try:
            response = await self.generate_with_retry(
                prompt="Health check",
                max_tokens=10,
                temperature=0,
                complexity=1
            )

            return {
                "status": "healthy" if response.text else "degraded",
                "latency_ms": response.latency_ms,
                "metrics": self.get_performance_metrics()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_performance_metrics()
            }

    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage"""
        return self.calculate_enterprise_cost(input_tokens, output_tokens)

    async def validate_connection(self) -> bool:
        """Validate provider connection"""
        health = await self.health_check()
        return health["status"] in ["healthy", "degraded"]