#!/usr/bin/env python3
"""
Model Registry CLI
Command-line interface for model registration and management
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from tabulate import tabulate

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_registry import ModelRegistry
from src.models.model_store import ModelStore
from src.models.model_metadata import (
    ModelMetadata, ModelTask, DeploymentStatus,
    TrainingConfig, DatasetInfo, PerformanceMetrics,
    HardwareRequirements, KhmerCapabilities
)


def register_model(args):
    """Register a new model"""
    registry = ModelRegistry(args.registry_path)

    # Load metadata from file if provided
    metadata = {}
    if args.metadata_file:
        with open(args.metadata_file, 'r') as f:
            if args.metadata_file.endswith('.yaml') or args.metadata_file.endswith('.yml'):
                metadata = yaml.safe_load(f)
            else:
                metadata = json.load(f)

    # Add command-line metadata
    if args.base_model:
        metadata['base_model'] = args.base_model
    if args.description:
        metadata['description'] = args.description
    if args.tags:
        metadata['tags'] = args.tags

    # Create performance metrics if provided
    performance_metrics = None
    if args.metrics:
        metrics_dict = {}
        for metric in args.metrics:
            key, value = metric.split('=')
            metrics_dict[key] = float(value)
        performance_metrics = PerformanceMetrics(**metrics_dict)

    # Register model
    model_id = registry.register(
        model_path=args.model_path,
        name=args.name,
        version=args.version,
        task=args.task,
        performance_metrics=performance_metrics,
        metadata=metadata
    )

    print(f"✓ Model registered successfully: {model_id}")

    # Show model info
    model = registry.get_model(model_id)
    if model:
        print("\n" + model.get_summary())


def list_models(args):
    """List registered models"""
    registry = ModelRegistry(args.registry_path)

    # Apply filters
    task = ModelTask(args.task) if args.task else None
    status = DeploymentStatus(args.status) if args.status else None

    models = registry.list_models(task=task, status=status, tags=args.tags)

    if not models:
        print("No models found")
        return

    # Prepare table data
    table_data = []
    for model in models:
        metrics_str = ""
        if model.performance_metrics:
            if model.performance_metrics.accuracy:
                metrics_str += f"Acc: {model.performance_metrics.accuracy:.3f} "
            if model.performance_metrics.khmer_bleu:
                metrics_str += f"BLEU: {model.performance_metrics.khmer_bleu:.3f}"

        table_data.append([
            model.model_id,
            model.name,
            model.version,
            model.task.value if model.task else "N/A",
            model.deployment_status.value if model.deployment_status else "N/A",
            metrics_str,
            model.updated_at.strftime("%Y-%m-%d")
        ])

    # Print table
    headers = ["Model ID", "Name", "Version", "Task", "Status", "Metrics", "Updated"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal models: {len(models)}")


def get_model(args):
    """Get model details"""
    registry = ModelRegistry(args.registry_path)

    model = registry.get_model(args.model_id)
    if not model:
        print(f"Model not found: {args.model_id}")
        sys.exit(1)

    if args.format == "json":
        print(model.to_json())
    elif args.format == "yaml":
        data = model.to_dict()
        print(yaml.dump(data, default_flow_style=False))
    else:
        # Human-readable format
        print("\n" + "="*60)
        print(model.get_summary())
        print("="*60)

        # Detailed information
        if model.training_config:
            print("\n📚 Training Configuration:")
            print(f"  Base Model: {model.training_config.base_model}")
            print(f"  Method: {model.training_config.method}")
            if model.training_config.epochs:
                print(f"  Epochs: {model.training_config.epochs}")
            if model.training_config.learning_rate:
                print(f"  Learning Rate: {model.training_config.learning_rate}")

        if model.hardware_requirements:
            print("\n💻 Hardware Requirements:")
            print(f"  Min GPU Memory: {model.hardware_requirements.min_gpu_memory_gb}GB")
            print(f"  Recommended GPU Memory: {model.hardware_requirements.recommended_gpu_memory_gb}GB")

        if model.khmer_capabilities:
            print("\n🇰🇭 Khmer Capabilities:")
            print(f"  Supports Khmer: {model.khmer_capabilities.supports_khmer}")
            if model.khmer_capabilities.specialized_domains:
                print(f"  Domains: {', '.join(model.khmer_capabilities.specialized_domains)}")

        if model.endpoints:
            print("\n🌐 Endpoints:")
            for endpoint in model.endpoints:
                print(f"  - {endpoint}")

        # Deployment history
        if args.show_history:
            history = registry.get_deployment_history(args.model_id)
            if history:
                print("\n📅 Deployment History:")
                for deployment in history:
                    print(f"  {deployment['deployed_at']}: {deployment['environment']} ({deployment['status']})")


def compare_models(args):
    """Compare two models"""
    registry = ModelRegistry(args.registry_path)

    comparison = registry.compare_versions(args.model1, args.model2)

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)

    print(f"\n📊 Model 1: {comparison['model1']['name']} (v{comparison['model1']['version']})")
    print(f"📊 Model 2: {comparison['model2']['name']} (v{comparison['model2']['version']})")

    # Metrics comparison
    if comparison['metrics_comparison']:
        print("\n📈 Metrics Comparison:")
        table_data = []
        for metric, values in comparison['metrics_comparison'].items():
            change_symbol = "↑" if values['difference'] > 0 else "↓" if values['difference'] < 0 else "="
            table_data.append([
                metric,
                f"{values['model1']:.4f}",
                f"{values['model2']:.4f}",
                f"{values['difference']:+.4f}",
                f"{change_symbol} {abs(values['pct_change']):.1f}%"
            ])

        headers = ["Metric", "Model 1", "Model 2", "Difference", "Change"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Improvements
    if comparison['improvements']:
        print("\n✅ Improvements:")
        for metric, change in comparison['improvements'].items():
            print(f"  - {metric}: {change:+.1f}%")

    # Regressions
    if comparison['regressions']:
        print("\n⚠️ Regressions:")
        for metric, change in comparison['regressions'].items():
            print(f"  - {metric}: {change:+.1f}%")


def update_metrics(args):
    """Update model metrics"""
    registry = ModelRegistry(args.registry_path)

    # Parse metrics
    metrics = {}
    for metric_str in args.metrics:
        key, value = metric_str.split('=')
        metrics[key] = float(value)

    registry.update_metrics(args.model_id, metrics)
    print(f"✓ Updated metrics for {args.model_id}")

    # Show updated model
    model = registry.get_model(args.model_id)
    if model and model.performance_metrics:
        print("\nUpdated metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def deploy_model(args):
    """Deploy model to environment"""
    registry = ModelRegistry(args.registry_path)

    status = DeploymentStatus(args.status) if args.status else DeploymentStatus.PRODUCTION

    registry.deploy_model(
        model_id=args.model_id,
        environment=args.environment,
        endpoint=args.endpoint,
        status=status
    )

    print(f"✓ Deployed {args.model_id} to {args.environment}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Status: {status.value}")


def search_models(args):
    """Search for models"""
    registry = ModelRegistry(args.registry_path)

    models = registry.search_models(args.query)

    if not models:
        print(f"No models found matching: {args.query}")
        return

    print(f"Found {len(models)} models matching '{args.query}':\n")

    for model in models:
        print(f"• {model.model_id}")
        print(f"  {model.name} v{model.version}")
        if model.description:
            print(f"  {model.description[:100]}...")
        print()


def get_statistics(args):
    """Get registry statistics"""
    registry = ModelRegistry(args.registry_path)
    store = ModelStore()

    registry_stats = registry.get_statistics()
    storage_stats = store.get_storage_stats()

    print("\n" + "="*60)
    print("Model Registry Statistics")
    print("="*60)

    print(f"\n📊 Models:")
    print(f"  Total: {registry_stats['total_models']}")

    if registry_stats['models_by_status']:
        print("\n  By Status:")
        for status, count in registry_stats['models_by_status'].items():
            print(f"    {status}: {count}")

    if registry_stats['models_by_task']:
        print("\n  By Task:")
        for task, count in registry_stats['models_by_task'].items():
            print(f"    {task}: {count}")

    print(f"\n💾 Storage:")
    print(f"  Total Size: {storage_stats['total_size_gb']:.2f} GB")
    print(f"  Checkpoints: {storage_stats['checkpoints']['count']} ({storage_stats['checkpoints']['size_gb']:.2f} GB)")
    print(f"  Adapters: {storage_stats['adapters']['count']} ({storage_stats['adapters']['size_gb']:.2f} GB)")
    print(f"  Quantized: {storage_stats['quantized']['count']} ({storage_stats['quantized']['size_gb']:.2f} GB)")

    if registry_stats['recent_models']:
        print("\n🕐 Recent Models:")
        for model in registry_stats['recent_models']:
            print(f"  - {model['name']} v{model['version']} ({model['updated_at'][:10]})")


def export_registry(args):
    """Export registry to file"""
    registry = ModelRegistry(args.registry_path)

    registry.export_registry(args.output)
    print(f"✓ Exported registry to {args.output}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Model Registry Management")
    parser.add_argument(
        "--registry-path",
        default="models/registry",
        help="Path to model registry"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new model")
    register_parser.add_argument("model_path", help="Path to model files")
    register_parser.add_argument("name", help="Model name")
    register_parser.add_argument("version", help="Model version")
    register_parser.add_argument("task", help="Model task (e.g., text-generation)")
    register_parser.add_argument("--base-model", help="Base model name")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--tags", nargs="+", help="Model tags")
    register_parser.add_argument("--metrics", nargs="+", help="Metrics (key=value)")
    register_parser.add_argument("--metadata-file", help="Metadata JSON/YAML file")
    register_parser.set_defaults(func=register_model)

    # List command
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--task", help="Filter by task")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_parser.set_defaults(func=list_models)

    # Get command
    get_parser = subparsers.add_parser("get", help="Get model details")
    get_parser.add_argument("model_id", help="Model ID")
    get_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text")
    get_parser.add_argument("--show-history", action="store_true", help="Show deployment history")
    get_parser.set_defaults(func=get_model)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model1", help="First model ID")
    compare_parser.add_argument("model2", help="Second model ID")
    compare_parser.set_defaults(func=compare_models)

    # Update metrics command
    metrics_parser = subparsers.add_parser("update-metrics", help="Update model metrics")
    metrics_parser.add_argument("model_id", help="Model ID")
    metrics_parser.add_argument("metrics", nargs="+", help="Metrics to update (key=value)")
    metrics_parser.set_defaults(func=update_metrics)

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model")
    deploy_parser.add_argument("model_id", help="Model ID")
    deploy_parser.add_argument("environment", help="Environment (dev, staging, production)")
    deploy_parser.add_argument("endpoint", help="Endpoint URL")
    deploy_parser.add_argument("--status", help="Deployment status")
    deploy_parser.set_defaults(func=deploy_model)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search models")
    search_parser.add_argument("query", help="Search query")
    search_parser.set_defaults(func=search_models)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")
    stats_parser.set_defaults(func=get_statistics)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export registry")
    export_parser.add_argument("output", help="Output file path")
    export_parser.set_defaults(func=export_registry)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()