"""
Khmer Model Evaluator
Comprehensive evaluation pipeline for Khmer language models
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field, asdict

from ..metrics.khmer_metrics import KhmerMetrics, KhmerMetricResult
from ..metrics.khmer_tokenizer import KhmerTokenizer
from ..providers.vllm_provider import VLLMProvider
from ..models.model_registry import ModelRegistry
from ..models.model_metadata import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run"""
    model_id: str
    dataset_path: str
    metrics_to_compute: List[str] = field(default_factory=lambda: ["all"])
    batch_size: int = 32
    max_samples: Optional[int] = None
    save_predictions: bool = True
    output_dir: str = "evaluation_results"
    temperature: float = 0.7
    max_tokens: int = 256
    num_gpus: int = 1
    device: str = "cuda"


@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    model_id: str
    dataset: str
    metrics: Dict[str, float]
    detailed_metrics: Dict[str, KhmerMetricResult]
    predictions: List[str]
    references: List[str]
    prompts: List[str]
    evaluation_time: float
    timestamp: str
    config: EvaluationConfig

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "detailed_metrics": {
                k: {"score": v.score, "details": v.details}
                for k, v in self.detailed_metrics.items()
            },
            "num_samples": len(self.predictions),
            "evaluation_time": self.evaluation_time,
            "timestamp": self.timestamp,
            "config": asdict(self.config)
        }

    def save(self, path: str):
        """Save results to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save main results as JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        # Save predictions if requested
        if self.config.save_predictions:
            predictions_path = path.with_suffix('.predictions.jsonl')
            with open(predictions_path, 'w', encoding='utf-8') as f:
                for prompt, pred, ref in zip(self.prompts, self.predictions, self.references):
                    f.write(json.dumps({
                        "prompt": prompt,
                        "prediction": pred,
                        "reference": ref
                    }, ensure_ascii=False) + '\n')

        logger.info(f"Results saved to {path}")


class KhmerEvaluator:
    """Main evaluator for Khmer language models"""

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        metrics_engine: Optional[KhmerMetrics] = None
    ):
        """
        Initialize evaluator

        Args:
            model_registry: Model registry instance
            metrics_engine: Khmer metrics instance
        """
        self.registry = model_registry or ModelRegistry()
        self.metrics = metrics_engine or KhmerMetrics()
        self.tokenizer = KhmerTokenizer()
        self.provider = None

    async def evaluate_model(
        self,
        config: EvaluationConfig
    ) -> EvaluationResult:
        """
        Run complete evaluation pipeline

        Args:
            config: Evaluation configuration

        Returns:
            Complete evaluation results
        """
        logger.info(f"Starting evaluation for model: {config.model_id}")
        start_time = time.time()

        # Load model
        logger.info("Loading model...")
        await self._load_model(config)

        # Load dataset
        logger.info("Loading dataset...")
        dataset = self._load_dataset(config.dataset_path, config.max_samples)

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = await self._generate_predictions(
            dataset["prompts"],
            config
        )

        # Calculate metrics
        logger.info("Computing metrics...")
        metrics_results = self._compute_metrics(
            predictions,
            dataset["references"],
            config.metrics_to_compute
        )

        # Create results
        evaluation_time = time.time() - start_time
        results = EvaluationResult(
            model_id=config.model_id,
            dataset=config.dataset_path,
            metrics={k: v.score for k, v in metrics_results.items()},
            detailed_metrics=metrics_results,
            predictions=predictions,
            references=dataset["references"],
            prompts=dataset["prompts"],
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat(),
            config=config
        )

        # Update model registry with metrics
        if self.registry:
            self._update_registry(config.model_id, results.metrics)

        # Save results
        output_path = Path(config.output_dir) / f"{config.model_id}_evaluation.json"
        results.save(output_path)

        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        return results

    async def _load_model(self, config: EvaluationConfig):
        """Load model for evaluation"""
        # Get model info from registry
        model_info = self.registry.get_model(config.model_id)
        if not model_info:
            raise ValueError(f"Model not found: {config.model_id}")

        # Initialize vLLM provider
        self.provider = VLLMProvider(
            model_name=model_info.model_path,
            device=config.device,
            num_gpus=config.num_gpus,
            temperature=config.temperature
        )

        # Load model
        await self.provider.load_model()

    async def _generate_predictions(
        self,
        prompts: List[str],
        config: EvaluationConfig
    ) -> List[str]:
        """Generate predictions for prompts"""
        predictions = []

        # Process in batches
        for i in tqdm(range(0, len(prompts), config.batch_size), desc="Generating"):
            batch = prompts[i:i + config.batch_size]

            # Generate batch predictions
            batch_results = await self.provider.batch_generate(
                batch,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )

            # Extract predictions
            for result in batch_results:
                if result.success:
                    predictions.append(result.content)
                else:
                    predictions.append("")
                    logger.warning(f"Generation failed: {result.error}")

        return predictions

    def _load_dataset(
        self,
        dataset_path: str,
        max_samples: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """Load evaluation dataset"""
        path = Path(dataset_path)

        if path.suffix == '.jsonl':
            # Load JSONL format
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        elif path.suffix == '.json':
            # Load JSON format
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.suffix == '.csv':
            # Load CSV format
            df = pd.read_csv(path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix}")

        # Extract prompts and references
        prompts = []
        references = []

        for item in data[:max_samples]:
            if isinstance(item, dict):
                # Handle different field names
                prompt = item.get('prompt') or item.get('input') or item.get('question')
                reference = item.get('reference') or item.get('output') or item.get('answer')

                if prompt and reference:
                    prompts.append(prompt)
                    references.append(reference)

        logger.info(f"Loaded {len(prompts)} samples from {dataset_path}")

        return {
            "prompts": prompts,
            "references": references
        }

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metrics_to_compute: List[str]
    ) -> Dict[str, KhmerMetricResult]:
        """Compute specified metrics"""
        if "all" in metrics_to_compute:
            return self.metrics.calculate_all_metrics(predictions, references)

        results = {}
        metric_map = {
            "bleu": lambda: self.metrics.calculate_khmer_bleu(
                predictions, references, "syllable"
            ),
            "character_accuracy": lambda: self.metrics.calculate_character_accuracy(
                predictions, references
            ),
            "syllable_f1": lambda: self.metrics.calculate_syllable_f1(
                predictions, references
            ),
            "word_segmentation": lambda: self.metrics.calculate_word_segmentation_accuracy(
                predictions, references
            ),
            "edit_distance": lambda: self.metrics.calculate_edit_distance(
                predictions, references, "char"
            ),
            "code_switching": lambda: self.metrics.calculate_code_switching_accuracy(
                predictions, references
            ),
            "khmer_errors": lambda: self.metrics.calculate_khmer_specific_errors(
                predictions, references
            )
        }

        for metric_name in metrics_to_compute:
            if metric_name in metric_map:
                results[metric_name] = metric_map[metric_name]()
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        return results

    def _update_registry(self, model_id: str, metrics: Dict[str, float]):
        """Update model registry with evaluation metrics"""
        try:
            # Prepare metrics for registry
            registry_metrics = {
                "khmer_bleu": metrics.get("bleu_syllable", 0.0),
                "character_accuracy": metrics.get("character_accuracy", 0.0),
                "syllable_f1": metrics.get("syllable_f1", 0.0)
            }

            self.registry.update_metrics(model_id, registry_metrics)
            logger.info(f"Updated registry for model {model_id}")
        except Exception as e:
            logger.error(f"Failed to update registry: {e}")

    async def compare_models(
        self,
        model_ids: List[str],
        dataset_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on same dataset

        Args:
            model_ids: List of model IDs to compare
            dataset_path: Path to evaluation dataset
            output_path: Optional path to save comparison

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_id in model_ids:
            logger.info(f"Evaluating model: {model_id}")

            config = EvaluationConfig(
                model_id=model_id,
                dataset_path=dataset_path
            )

            eval_result = await self.evaluate_model(config)

            # Collect metrics
            model_results = {
                "model_id": model_id,
                "evaluation_time": eval_result.evaluation_time,
                **eval_result.metrics
            }
            results.append(model_results)

        # Create comparison dataframe
        df = pd.DataFrame(results)
        df = df.set_index("model_id")

        # Sort by overall score (average of all metrics)
        metric_cols = [c for c in df.columns if c != "evaluation_time"]
        df["overall_score"] = df[metric_cols].mean(axis=1)
        df = df.sort_values("overall_score", ascending=False)

        # Save if requested
        if output_path:
            df.to_csv(output_path)
            logger.info(f"Comparison saved to {output_path}")

        return df

    def create_leaderboard(
        self,
        evaluation_dir: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create leaderboard from evaluation results

        Args:
            evaluation_dir: Directory with evaluation results
            output_path: Optional path to save leaderboard

        Returns:
            DataFrame with leaderboard
        """
        eval_dir = Path(evaluation_dir)
        results = []

        # Load all evaluation results
        for result_file in eval_dir.glob("*_evaluation.json"):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    "model_id": data["model_id"],
                    "dataset": data["dataset"],
                    "num_samples": data["num_samples"],
                    **data["metrics"]
                })

        # Create leaderboard
        df = pd.DataFrame(results)

        if not df.empty:
            # Calculate overall score
            metric_cols = [c for c in df.columns if c not in ["model_id", "dataset", "num_samples"]]
            df["overall_score"] = df[metric_cols].mean(axis=1)

            # Rank models
            df["rank"] = df["overall_score"].rank(ascending=False, method="min").astype(int)
            df = df.sort_values("rank")

            # Save if requested
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Leaderboard saved to {output_path}")

        return df


async def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Khmer language models")
    parser.add_argument("--model-id", required=True, help="Model ID from registry")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset")
    parser.add_argument("--metrics", nargs="+", default=["all"], help="Metrics to compute")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")

    args = parser.parse_args()

    # Create config
    config = EvaluationConfig(
        model_id=args.model_id,
        dataset_path=args.dataset,
        metrics_to_compute=args.metrics,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )

    # Run evaluation
    evaluator = KhmerEvaluator()
    results = await evaluator.evaluate_model(config)

    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, score in results.metrics.items():
        print(f"{metric}: {score:.4f}")
    print(f"\nTotal time: {results.evaluation_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())