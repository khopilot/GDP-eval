#!/usr/bin/env python3
"""
Model Comparison Tool for GDP Evaluation
Compare performance between different models (baseline vs fine-tuned)
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.providers.grok_provider import GrokProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.huggingface_provider import HuggingFaceProvider
from src.providers.vllm_provider import VLLMProvider
from src.core.provider_evaluator import Evaluator
from src.tasks.task_converter import TaskConverter
from src.metrics.performance_metrics import PerformanceAnalyzer


class ModelComparator:
    """Compare multiple models on same tasks"""

    def __init__(self):
        self.converter = TaskConverter()
        self.results = {}

    async def evaluate_model(
        self,
        provider,
        model_name: str,
        tasks: List[Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on tasks

        Args:
            provider: LLM provider instance
            model_name: Name for identification
            tasks: List of evaluation tasks

        Returns:
            Evaluation results
        """
        print(f"\nEvaluating {model_name}...")
        print("=" * 60)

        evaluator = Evaluator(
            provider=provider,
            output_dir=f"results/comparison/{model_name}"
        )

        analyzer = PerformanceAnalyzer()
        task_results = []

        # Run each task
        for i, task in enumerate(tasks, 1):
            print(f"Task {i}/{len(tasks)}: {task.occupation} ({task.category})")

            result = await evaluator.evaluate_single(
                task,
                max_tokens=1024,
                temperature=0.7
            )

            task_results.append(result)
            analyzer.add_result(result)

            if result.success:
                print(f"  ✓ Success in {result.latency_ms:.0f}ms")
            else:
                print(f"  ✗ Failed: {result.error}")

        # Calculate metrics
        stats = analyzer.get_statistics()

        summary = {
            "model": model_name,
            "provider": provider.provider_name,
            "timestamp": datetime.now().isoformat(),
            "tasks_total": len(tasks),
            "tasks_successful": stats["successful_tasks"],
            "tasks_failed": stats["failed_tasks"],
            "success_rate": stats["success_rate"],
            "avg_latency_ms": stats["latency"].get("mean", 0),
            "total_tokens": stats["tokens"].get("total", 0),
            "total_cost": stats["cost"].get("total", 0),
            "efficiency_score": analyzer.get_efficiency_score(),
            "category_performance": analyzer.get_category_performance(),
            "results": [r.to_dict() for r in task_results]
        }

        return summary

    async def compare_models(
        self,
        model_configs: List[Dict[str, Any]],
        task_limit: int = None
    ):
        """
        Compare multiple models

        Args:
            model_configs: List of model configurations
            task_limit: Limit number of tasks (None for all)
        """
        print("\n" + "=" * 80)
        print("GDP-eval Model Comparison")
        print("=" * 80)

        # Load test tasks
        print("\nLoading professional tasks...")
        tasks = self.converter.create_cambodia_test_suite()

        if task_limit:
            tasks = tasks[:task_limit]

        print(f"Loaded {len(tasks)} tasks for evaluation")

        # Evaluate each model
        for config in model_configs:
            provider = self._create_provider(config)
            if provider:
                results = await self.evaluate_model(
                    provider,
                    config["name"],
                    tasks
                )
                self.results[config["name"]] = results

                # Close provider if needed
                if hasattr(provider, 'close'):
                    await provider.close()

        # Generate comparison report
        self._generate_comparison_report()

    def _create_provider(self, config: Dict[str, Any]):
        """Create provider from configuration"""
        provider_type = config["type"]

        if provider_type == "grok":
            return GrokProvider(
                api_key=config["api_key"],
                model=config.get("model", "grok-3"),
                timeout=config.get("timeout", 120)
            )
        elif provider_type == "ollama":
            return OllamaProvider(
                model=config["model"],
                base_url=config.get("base_url", "http://localhost:11434")
            )
        elif provider_type == "huggingface":
            return HuggingFaceProvider(
                model_name=config["model"],
                device=config.get("device", "cuda")
            )
        elif provider_type == "vllm":
            return VLLMProvider(
                model=config["model"],
                tensor_parallel_size=config.get("tensor_parallel_size", 1)
            )
        else:
            print(f"Unknown provider type: {provider_type}")
            return None

    def _generate_comparison_report(self):
        """Generate comparison report"""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        # Create comparison table
        print("\n| Model | Success Rate | Avg Latency | Tokens | Cost | Efficiency |")
        print("|-------|--------------|-------------|--------|------|------------|")

        for name, result in self.results.items():
            print(f"| {name:<15} "
                  f"| {result['success_rate']*100:>6.1f}% "
                  f"| {result['avg_latency_ms']:>8.0f}ms "
                  f"| {result['total_tokens']:>6.0f} "
                  f"| ${result['total_cost']:>6.4f} "
                  f"| {result['efficiency_score']:>6.1f}% |")

        # Calculate improvements
        if len(self.results) == 2:
            models = list(self.results.keys())
            baseline = self.results[models[0]]
            improved = self.results[models[1]]

            print("\n" + "=" * 80)
            print(f"IMPROVEMENT: {models[1]} vs {models[0]}")
            print("=" * 80)

            improvements = {
                "Success Rate": (improved['success_rate'] - baseline['success_rate']) * 100,
                "Latency": baseline['avg_latency_ms'] - improved['avg_latency_ms'],
                "Efficiency": improved['efficiency_score'] - baseline['efficiency_score']
            }

            for metric, value in improvements.items():
                sign = "+" if value > 0 else ""
                print(f"{metric}: {sign}{value:.1f}")

        # Save detailed results
        output_dir = Path("results/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "models_compared": list(self.results.keys()),
                "task_count": self.results[list(self.results.keys())[0]]["tasks_total"],
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")


async def main():
    """Main comparison function"""

    # Configuration for models to compare
    model_configs = [
        {
            "name": "grok-baseline",
            "type": "grok",
            "api_key": "YOUR-API-KEY-HERE",
            "model": "grok-3",
            "timeout": 120
        },
        # Add your fine-tuned model here when ready:
        # {
        #     "name": "khmer-finetuned-v1",
        #     "type": "ollama",
        #     "model": "khmer-llama-7b",
        #     "base_url": "http://localhost:11434"
        # }
    ]

    # Run comparison
    comparator = ModelComparator()
    await comparator.compare_models(
        model_configs,
        task_limit=3  # Use 3 tasks for quick testing, None for all
    )


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           GDP-eval Model Comparison Tool                     ║
    ║                                                              ║
    ║  Compare baseline vs fine-tuned models on professional tasks ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())