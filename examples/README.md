# Examples

This directory contains example scripts and usage demonstrations.

## Contents

- `demo.py` - Interactive demo of the GDP evaluation framework

## Running Examples

### Basic Demo
```bash
python examples/demo.py
```

## Example Use Cases

### 1. Evaluate a Local Model
```python
from src.evaluation import KhmerEvaluator, EvaluationConfig

config = EvaluationConfig(
    model_id="my-model-v1",
    dataset_path="data/evaluation/khmer_test_dataset.jsonl",
    metrics_to_compute=["all"]
)

evaluator = KhmerEvaluator()
results = await evaluator.evaluate_model(config)
print(f"BLEU Score: {results.metrics['bleu_syllable']:.4f}")
```

### 2. Compare Multiple Models
```python
evaluator = KhmerEvaluator()
df = await evaluator.compare_models(
    model_ids=["model-v1", "model-v2", "model-v3"],
    dataset_path="data/evaluation/khmer_test_dataset.jsonl"
)
print(df)
```

### 3. Register and Deploy a Model
```python
from src.models import ModelRegistry

registry = ModelRegistry()
model_id = registry.register(
    model_path="path/to/model",
    name="Khmer-GPT",
    version="1.0.0",
    task="text-generation"
)

registry.deploy_model(
    model_id=model_id,
    environment="production",
    endpoint="http://localhost:8000"
)
```

## Creating Your Own Examples

Feel free to add your own example scripts to this directory. Follow the naming convention:
- `example_*.py` for standalone examples
- `test_*.py` for test scripts
- `benchmark_*.py` for performance benchmarks