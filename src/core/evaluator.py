"""
Model Evaluator for GDPval Framework
Handles evaluation of language models on tasks
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from src.core.task_loader import EvaluationTask

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a model on a task"""
    task_id: str
    model_name: str
    response: str
    latency_ms: float
    api_cost: float
    timestamp: str
    metadata: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(self)


class BaseEvaluator:
    """Base class for model evaluators"""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize evaluator

        Args:
            model_name: Name/identifier of the model
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.config = kwargs
        self.results: List[EvaluationResult] = []

    async def evaluate_task(self, task: EvaluationTask) -> EvaluationResult:
        """
        Evaluate a single task

        Args:
            task: Task to evaluate

        Returns:
            Evaluation result
        """
        start_time = time.time()
        error = None
        response = ""
        api_cost = 0.0

        try:
            # Prepare context
            context = self._prepare_context(task)

            # Generate response
            response = await self._generate_response(context)

            # Calculate cost
            api_cost = self._calculate_cost(context, response)

        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000

        result = EvaluationResult(
            task_id=task.task_id,
            model_name=self.model_name,
            response=response,
            latency_ms=latency_ms,
            api_cost=api_cost,
            timestamp=datetime.now().isoformat(),
            metadata={
                "category": task.category,
                "occupation": task.occupation,
                "difficulty": task.metadata.get("difficulty_level", 0)
            },
            error=error
        )

        self.results.append(result)
        return result

    async def evaluate_batch(self, tasks: List[EvaluationTask], max_concurrent: int = 5) -> List[EvaluationResult]:
        """
        Evaluate multiple tasks concurrently

        Args:
            tasks: List of tasks to evaluate
            max_concurrent: Maximum concurrent evaluations

        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(task):
            async with semaphore:
                return await self.evaluate_task(task)

        results = await asyncio.gather(
            *[evaluate_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].task_id} failed: {result}")
                processed_results.append(
                    EvaluationResult(
                        task_id=tasks[i].task_id,
                        model_name=self.model_name,
                        response="",
                        latency_ms=0,
                        api_cost=0,
                        timestamp=datetime.now().isoformat(),
                        metadata={},
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _prepare_context(self, task: EvaluationTask) -> str:
        """
        Prepare context/prompt for the model

        Args:
            task: Evaluation task

        Returns:
            Formatted context string
        """
        # Build context from task components
        context_parts = []

        # Add role/context if available
        if 'context' in task.prompt:
            context_parts.append(f"Context: {task.prompt['context']}")

        # Add main instruction
        if 'instruction' in task.prompt:
            context_parts.append(f"Task: {task.prompt['instruction']}")

        # Add output format requirements
        if 'output_format' in task.prompt:
            context_parts.append(f"Output Format: {task.prompt['output_format']}")

        # Add reference to files if needed
        if task.reference_files:
            file_list = ", ".join([f['file_name'] for f in task.reference_files])
            context_parts.append(f"Reference Files: {file_list}")

        return "\n\n".join(context_parts)

    async def _generate_response(self, context: str) -> str:
        """
        Generate response from model (to be implemented by subclasses)

        Args:
            context: Input context/prompt

        Returns:
            Model response
        """
        raise NotImplementedError("Subclasses must implement _generate_response")

    def _calculate_cost(self, context: str, response: str) -> float:
        """
        Calculate API cost for the request

        Args:
            context: Input context
            response: Model response

        Returns:
            Estimated cost in USD
        """
        # Default implementation (can be overridden)
        # Estimate tokens (rough approximation)
        input_tokens = len(context.split()) * 1.3
        output_tokens = len(response.split()) * 1.3

        # Use default pricing (can be customized per model)
        input_price_per_1k = self.config.get('input_price_per_1k', 0.0001)
        output_price_per_1k = self.config.get('output_price_per_1k', 0.0002)

        cost = (input_tokens * input_price_per_1k + output_tokens * output_price_per_1k) / 1000
        return cost

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.results:
            return {"total_evaluations": 0}

        successful_results = [r for r in self.results if not r.error]
        failed_results = [r for r in self.results if r.error]

        return {
            "total_evaluations": len(self.results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "avg_latency_ms": sum(r.latency_ms for r in successful_results) / len(successful_results) if successful_results else 0,
            "total_cost": sum(r.api_cost for r in self.results),
            "error_rate": len(failed_results) / len(self.results) if self.results else 0,
            "categories_evaluated": list(set(r.metadata.get('category', 'unknown') for r in self.results))
        }


class KhmerModelEvaluator(BaseEvaluator):
    """Evaluator specifically for Khmer language models"""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize Khmer model evaluator

        Args:
            model_name: Name of the model
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        self.khmer_support = kwargs.get('khmer_support', True)
        self.bilingual_mode = kwargs.get('bilingual_mode', True)

    def _prepare_context(self, task: EvaluationTask) -> str:
        """
        Prepare context with Khmer language considerations

        Args:
            task: Evaluation task

        Returns:
            Formatted context string
        """
        context_parts = []

        # Use Khmer or bilingual context based on configuration
        if self.khmer_support and 'context' in task.prompt:
            context_parts.append(f"បរិបទ: {task.prompt['context']}")
        elif 'context_english' in task.prompt:
            context_parts.append(f"Context: {task.prompt['context_english']}")

        # Add instruction in appropriate language
        if self.khmer_support and 'instruction' in task.prompt:
            context_parts.append(f"ការណែនាំ: {task.prompt['instruction']}")
        elif 'instruction_english' in task.prompt:
            context_parts.append(f"Instruction: {task.prompt['instruction_english']}")

        # Add bilingual note if needed
        if self.bilingual_mode:
            context_parts.append("Please provide your response in Khmer with technical terms in English where appropriate.")

        return "\n\n".join(context_parts)

    def validate_khmer_response(self, response: str) -> Dict[str, Any]:
        """
        Validate that response contains appropriate Khmer content

        Args:
            response: Model response

        Returns:
            Validation results
        """
        khmer_char_count = sum(1 for char in response if 0x1780 <= ord(char) <= 0x17FF)
        total_chars = len(response)
        khmer_percentage = (khmer_char_count / total_chars * 100) if total_chars > 0 else 0

        return {
            "contains_khmer": khmer_char_count > 0,
            "khmer_percentage": khmer_percentage,
            "khmer_char_count": khmer_char_count,
            "total_chars": total_chars,
            "is_sufficient_khmer": khmer_percentage > 30  # At least 30% Khmer content
        }


class BaselineEvaluator(BaseEvaluator):
    """Simple baseline evaluator for comparison"""

    async def _generate_response(self, context: str) -> str:
        """
        Generate a simple baseline response

        Args:
            context: Input context

        Returns:
            Baseline response
        """
        # Simple rule-based response
        if "financial" in context.lower() or "ហិរញ្ញវត្ថុ" in context:
            return "Based on the financial data provided, the analysis shows positive growth trends with manageable risk levels. Further optimization of interest rates and credit portfolio diversification is recommended."
        elif "technology" in context.lower() or "បច្ចេកវិទ្យា" in context:
            return "The technical solution involves implementing modern best practices with appropriate tools and frameworks to achieve optimal performance and maintainability."
        elif "healthcare" in context.lower() or "សុខភាព" in context:
            return "The healthcare assessment indicates standard protocols should be followed with regular monitoring and preventive measures to ensure optimal patient outcomes."
        else:
            return "The task has been analyzed according to the specified requirements. The solution addresses all key aspects mentioned in the instruction."

    def _calculate_cost(self, context: str, response: str) -> float:
        """
        Baseline has no API cost

        Args:
            context: Input context
            response: Model response

        Returns:
            Zero cost
        """
        return 0.0