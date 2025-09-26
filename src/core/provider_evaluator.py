"""
Provider-based Evaluator for GDP Framework
Bridges LLM providers with evaluation tasks
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json

from src.core.task_loader import EvaluationTask
from src.providers.base_provider import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a task"""
    task_id: str
    model_name: str
    provider_name: str
    response: str
    latency_ms: float
    tokens_used: int
    api_cost: float
    success: bool
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(self)


class Evaluator:
    """
    Main evaluator that works with any LLM provider
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        output_dir: str = "results",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluator with provider

        Args:
            provider: LLM provider instance
            output_dir: Directory for results
            config: Additional configuration
        """
        self.provider = provider
        self.output_dir = output_dir
        self.config = config or {}
        self.results: List[EvaluationResult] = []

    async def evaluate_single(
        self,
        task: EvaluationTask,
        **generation_params
    ) -> EvaluationResult:
        """
        Evaluate a single task

        Args:
            task: Evaluation task
            **generation_params: Parameters for generation

        Returns:
            Evaluation result
        """
        # Prepare prompt
        prompt = self._prepare_prompt(task)

        # Set default generation parameters
        params = {
            "max_tokens": 512,
            "temperature": 0.7,
            **self.config,
            **generation_params
        }

        # Add system prompt if needed
        system_prompt = None
        if task.metadata.get("language") == "khmer":
            system_prompt = "You are a helpful assistant fluent in Khmer. Respond in Khmer with English technical terms where appropriate."
        elif task.metadata.get("language") == "bilingual":
            system_prompt = "You are a bilingual assistant fluent in both Khmer and English. Provide responses mixing both languages naturally."

        if system_prompt:
            params["system_prompt"] = system_prompt

        # Generate response
        start_time = time.time()
        try:
            response = await self.provider.generate(prompt, **params)
            latency_ms = (time.time() - start_time) * 1000

            # Check if response has text (successful)
            if response.text:
                result = EvaluationResult(
                    task_id=task.task_id,
                    model_name=response.model,
                    provider_name=response.provider,
                    response=response.text,  # Use text field
                    latency_ms=latency_ms,
                    tokens_used=response.tokens_used,
                    api_cost=response.cost,  # Use cost from response
                    success=True,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "category": task.category,
                        "occupation": task.occupation,
                        "language": task.metadata.get("language", "english"),
                        "complexity": task.metadata.get("complexity", 1)
                    }
                )
            else:
                result = EvaluationResult(
                    task_id=task.task_id,
                    model_name=response.model,
                    provider_name=response.provider,
                    response="",
                    latency_ms=latency_ms,
                    tokens_used=0,
                    api_cost=0,
                    success=False,
                    timestamp=datetime.now().isoformat(),
                    metadata={"category": task.category},
                    error=response.error
                )

        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            result = EvaluationResult(
                task_id=task.task_id,
                model_name=self.provider.model_name,
                provider_name=self.provider.provider_name,
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                api_cost=0,
                success=False,
                timestamp=datetime.now().isoformat(),
                metadata={"category": task.category},
                error=str(e)
            )

        self.results.append(result)
        return result

    async def evaluate_batch(
        self,
        tasks: List[EvaluationTask],
        max_concurrent: int = 5,
        **generation_params
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple tasks concurrently

        Args:
            tasks: List of tasks
            max_concurrent: Maximum concurrent evaluations
            **generation_params: Parameters for generation

        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(task):
            async with semaphore:
                return await self.evaluate_single(task, **generation_params)

        results = await asyncio.gather(
            *[evaluate_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].task_id} failed: {result}")
                processed_results.append(
                    EvaluationResult(
                        task_id=tasks[i].task_id,
                        model_name=self.provider.model_name,
                        provider_name=self.provider.provider_name,
                        response="",
                        latency_ms=0,
                        tokens_used=0,
                        api_cost=0,
                        success=False,
                        timestamp=datetime.now().isoformat(),
                        metadata={"category": tasks[i].category},
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _prepare_prompt(self, task: EvaluationTask) -> str:
        """
        Prepare prompt from task

        Args:
            task: Evaluation task

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        # Add context
        if isinstance(task.prompt, dict):
            if "context" in task.prompt:
                prompt_parts.append(f"Context: {task.prompt['context']}")

            # Add instruction
            if "instruction" in task.prompt:
                prompt_parts.append(f"Task: {task.prompt['instruction']}")

            # Add requirements
            if "requirements" in task.prompt:
                requirements = task.prompt["requirements"]
                if isinstance(requirements, list):
                    prompt_parts.append("Requirements:")
                    for req in requirements:
                        prompt_parts.append(f"- {req}")
                else:
                    prompt_parts.append(f"Requirements: {requirements}")

            # Add output format
            if "output_format" in task.prompt:
                prompt_parts.append(f"Output Format: {task.prompt['output_format']}")
        else:
            # Simple string prompt
            prompt_parts.append(str(task.prompt))

        # Add file references if any
        if task.reference_files:
            files = [f["file_name"] for f in task.reference_files]
            prompt_parts.append(f"Reference Files: {', '.join(files)}")

        return "\n\n".join(prompt_parts)

    def _calculate_cost(self, tokens: int) -> float:
        """
        Calculate API cost based on tokens

        Args:
            tokens: Number of tokens used

        Returns:
            Estimated cost in USD
        """
        # Default pricing (can be overridden in config)
        price_per_1k = self.config.get("price_per_1k_tokens", 0.001)
        return (tokens / 1000) * price_per_1k

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics

        Returns:
            Statistics dictionary
        """
        if not self.results:
            return {"total": 0}

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        stats = {
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results),
            "latency": {
                "mean": sum(r.latency_ms for r in successful) / len(successful) if successful else 0,
                "min": min((r.latency_ms for r in successful), default=0),
                "max": max((r.latency_ms for r in successful), default=0)
            },
            "tokens": {
                "total": sum(r.tokens_used for r in successful),
                "mean": sum(r.tokens_used for r in successful) / len(successful) if successful else 0
            },
            "cost": {
                "total": sum(r.api_cost for r in self.results),
                "mean": sum(r.api_cost for r in self.results) / len(self.results) if self.results else 0
            },
            "by_category": self._get_category_stats(),
            "errors": [{"task_id": r.task_id, "error": r.error} for r in failed]
        }

        return stats

    def _get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by category"""
        category_stats = {}

        for result in self.results:
            category = result.metadata.get("category", "unknown")
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_latency": []
                }

            category_stats[category]["total"] += 1
            if result.success:
                category_stats[category]["successful"] += 1
                category_stats[category]["avg_latency"].append(result.latency_ms)
            else:
                category_stats[category]["failed"] += 1

        # Calculate averages
        for category, stats in category_stats.items():
            latencies = stats.pop("avg_latency")
            stats["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0

        return category_stats

    def save_results(self, filepath: Optional[str] = None):
        """
        Save evaluation results to file

        Args:
            filepath: Path to save results
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.output_dir}/evaluation_{timestamp}.json"

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        output = {
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "results": [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {filepath}")
        return filepath