#!/usr/bin/env python3
"""
Khmer Model Evaluation CLI
Command-line interface for evaluating Khmer language models
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional
import json
import pandas as pd
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.khmer_evaluator import KhmerEvaluator, EvaluationConfig
from src.models.model_registry import ModelRegistry
from src.metrics.khmer_metrics import KhmerMetrics


def run_evaluation(args):
    """Run model evaluation"""

    # Create config
    config = EvaluationConfig(
        model_id=args.model_id,
        dataset_path=args.dataset,
        metrics_to_compute=args.metrics,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_gpus=args.num_gpus,
        device=args.device,
        save_predictions=args.save_predictions
    )

    # Run evaluation
    evaluator = KhmerEvaluator()
    results = asyncio.run(evaluator.evaluate_model(config))

    # Print results
    print("\n" + "="*60)
    print("Khmer Model Evaluation Results")
    print("="*60)
    print(f"\nModel: {results.model_id}")
    print(f"Dataset: {results.dataset}")
    print(f"Samples: {len(results.predictions)}")
    print(f"Time: {results.evaluation_time:.2f} seconds")

    print("\n📊 Metrics:")
    print("-" * 40)

    # Format metrics for display
    metric_data = []
    for name, score in results.metrics.items():
        # Get detailed info if available
        if name in results.detailed_metrics:
            details = results.detailed_metrics[name].details
            std = details.get('std_dev', 0)
            metric_data.append([name, f"{score:.4f}", f"±{std:.4f}"])
        else:
            metric_data.append([name, f"{score:.4f}", ""])

    headers = ["Metric", "Score", "Std Dev"]
    print(tabulate(metric_data, headers=headers, tablefmt="simple"))

    if args.show_details:
        print("\n📈 Detailed Metrics:")
        print("-" * 40)
        for name, result in results.detailed_metrics.items():
            print(f"\n{result.metric_name}:")
            for key, value in result.details.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    if args.show_examples and len(results.predictions) > 0:
        print("\n📝 Example Predictions:")
        print("-" * 40)
        num_examples = min(args.show_examples, len(results.predictions))
        for i in range(num_examples):
            print(f"\nExample {i+1}:")
            print(f"Prompt: {results.prompts[i]}")
            print(f"Reference: {results.references[i]}")
            print(f"Prediction: {results.predictions[i]}")

    print("\n✓ Evaluation complete!")
    print(f"Results saved to: {args.output_dir}/{args.model_id}_evaluation.json")


def compare_models(args):
    """Compare multiple models"""

    evaluator = KhmerEvaluator()

    # Run comparison
    df = asyncio.run(evaluator.compare_models(
        model_ids=args.model_ids,
        dataset_path=args.dataset,
        output_path=args.output
    ))

    # Print comparison
    print("\n" + "="*60)
    print("Model Comparison Results")
    print("="*60)

    print("\n📊 Performance Comparison:")
    print(df.to_string())

    # Find best model for each metric
    print("\n🏆 Best Models by Metric:")
    print("-" * 40)
    for col in df.columns:
        if col not in ["evaluation_time", "overall_score"]:
            best_model = df[col].idxmax()
            best_score = df[col].max()
            print(f"{col}: {best_model} ({best_score:.4f})")

    # Overall winner
    print("\n🥇 Overall Best Model:")
    best_overall = df["overall_score"].idxmax()
    best_overall_score = df["overall_score"].max()
    print(f"{best_overall} (score: {best_overall_score:.4f})")

    if args.output:
        print(f"\n✓ Comparison saved to: {args.output}")


def create_leaderboard(args):
    """Create leaderboard from evaluations"""

    evaluator = KhmerEvaluator()

    # Create leaderboard
    df = evaluator.create_leaderboard(
        evaluation_dir=args.eval_dir,
        output_path=args.output
    )

    if df.empty:
        print("No evaluation results found")
        return

    # Print leaderboard
    print("\n" + "="*60)
    print("Khmer Model Leaderboard")
    print("="*60)

    # Display columns
    display_cols = ["rank", "model_id", "overall_score"]

    # Add key metrics if available
    metric_cols = [c for c in df.columns if c not in ["rank", "model_id", "dataset", "num_samples", "overall_score"]]
    display_cols.extend(metric_cols[:3])  # Show top 3 metrics

    print("\n🏆 Rankings:")
    print(df[display_cols].to_string(index=False))

    # Statistics
    print("\n📈 Statistics:")
    print("-" * 40)
    print(f"Total models: {len(df)}")
    print(f"Average score: {df['overall_score'].mean():.4f}")
    print(f"Score range: {df['overall_score'].min():.4f} - {df['overall_score'].max():.4f}")

    if args.output:
        print(f"\n✓ Leaderboard saved to: {args.output}")


def test_metrics(args):
    """Test metrics on sample texts"""

    metrics = KhmerMetrics()

    # Sample texts for testing
    if args.text1 and args.text2:
        predictions = [args.text1]
        references = [args.text2]
    else:
        # Default test samples
        predictions = [
            "ធនាគារជាតិ​នៃ​កម្ពុជា​គ្រប់គ្រង​គោលនយោបាយ​រូបិយវត្ថុ",
            "GDP means ផលិតផល​ក្នុង​ស្រុក​សរុប"
        ]
        references = [
            "ធនាគារជាតិ​នៃ​កម្ពុជា​មាន​តួនាទី​គ្រប់គ្រង​គោលនយោបាយ​រូបិយវត្ថុ",
            "GDP ជា​ភាសា​ខ្មែរ​គឺ ផលិតផល​ក្នុង​ស្រុក​សរុប"
        ]

    print("\n" + "="*60)
    print("Khmer Metrics Test")
    print("="*60)

    # Test each metric
    if "all" in args.metrics or "bleu" in args.metrics:
        print("\n📊 BLEU Scores:")
        for tokenization in ["char", "syllable", "word"]:
            result = metrics.calculate_khmer_bleu(predictions, references, tokenization)
            print(f"  {tokenization}: {result.score:.4f}")

    if "all" in args.metrics or "accuracy" in args.metrics:
        result = metrics.calculate_character_accuracy(predictions, references)
        print(f"\n📊 Character Accuracy: {result.score:.4f}")

    if "all" in args.metrics or "syllable" in args.metrics:
        result = metrics.calculate_syllable_f1(predictions, references)
        print(f"\n📊 Syllable F1: {result.score:.4f}")
        print(f"  Precision: {result.details['precision']:.4f}")
        print(f"  Recall: {result.details['recall']:.4f}")

    if "all" in args.metrics or "segmentation" in args.metrics:
        result = metrics.calculate_word_segmentation_accuracy(predictions, references)
        print(f"\n📊 Word Segmentation: {result.score:.4f}")

    if "all" in args.metrics or "edit" in args.metrics:
        print("\n📊 Edit Distance (similarity):")
        for level in ["char", "syllable", "word"]:
            result = metrics.calculate_edit_distance(predictions, references, level)
            print(f"  {level}: {result.score:.4f}")

    if "all" in args.metrics or "errors" in args.metrics:
        result = metrics.calculate_khmer_specific_errors(predictions, references)
        print(f"\n📊 Khmer-Specific Accuracy: {result.score:.4f}")
        print(f"  Subscript errors: {result.details['avg_subscript_errors']:.2f}")
        print(f"  Vowel errors: {result.details['avg_vowel_errors']:.2f}")
        print(f"  Sign errors: {result.details['avg_sign_errors']:.2f}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Khmer Model Evaluation CLI")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single model")
    eval_parser.add_argument("model_id", help="Model ID from registry")
    eval_parser.add_argument("dataset", help="Path to evaluation dataset")
    eval_parser.add_argument("--metrics", nargs="+", default=["all"],
                            help="Metrics to compute")
    eval_parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for generation")
    eval_parser.add_argument("--max-samples", type=int,
                            help="Maximum samples to evaluate")
    eval_parser.add_argument("--output-dir", default="evaluation_results",
                            help="Output directory")
    eval_parser.add_argument("--temperature", type=float, default=0.7,
                            help="Generation temperature")
    eval_parser.add_argument("--max-tokens", type=int, default=256,
                            help="Maximum tokens to generate")
    eval_parser.add_argument("--num-gpus", type=int, default=1,
                            help="Number of GPUs to use")
    eval_parser.add_argument("--device", default="cuda",
                            help="Device (cuda/cpu)")
    eval_parser.add_argument("--save-predictions", action="store_true",
                            help="Save predictions to file")
    eval_parser.add_argument("--show-details", action="store_true",
                            help="Show detailed metrics")
    eval_parser.add_argument("--show-examples", type=int,
                            help="Show N example predictions")
    eval_parser.set_defaults(func=run_evaluation)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("model_ids", nargs="+", help="Model IDs to compare")
    compare_parser.add_argument("dataset", help="Path to evaluation dataset")
    compare_parser.add_argument("--output", help="Output CSV file")
    compare_parser.set_defaults(func=compare_models)

    # Leaderboard command
    board_parser = subparsers.add_parser("leaderboard", help="Create leaderboard")
    board_parser.add_argument("eval_dir", help="Directory with evaluation results")
    board_parser.add_argument("--output", help="Output CSV file")
    board_parser.set_defaults(func=create_leaderboard)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test metrics on sample texts")
    test_parser.add_argument("--text1", help="First text (prediction)")
    test_parser.add_argument("--text2", help="Second text (reference)")
    test_parser.add_argument("--metrics", nargs="+", default=["all"],
                            help="Metrics to test")
    test_parser.set_defaults(func=test_metrics)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()