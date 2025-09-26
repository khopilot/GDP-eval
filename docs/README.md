# Documentation

## Contents

- `gdpval-khmer-readme.md` - Original Khmer GDP evaluation framework documentation
- `github-workflow.txt` - GitHub Actions workflow reference

## Key Documentation

### Architecture Overview
The GDP-eval framework is designed for evaluating Khmer language models with a focus on economic and financial domains.

### Main Components

1. **Model Providers** - Support for vLLM, HuggingFace, Ollama, and local models
2. **Model Registry** - Centralized model management with versioning
3. **Khmer Metrics** - Specialized evaluation metrics for Khmer text
4. **Evaluation Pipeline** - Comprehensive model evaluation system

### Quick Start Guide

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Register a model:
```bash
python scripts/register_model.py register path/to/model "Model Name" "1.0" "text-generation"
```

3. Run evaluation:
```bash
python scripts/evaluate_khmer_model.py evaluate model-v1.0 data/evaluation/khmer_test_dataset.jsonl
```

### API Documentation
See individual module docstrings for detailed API documentation.

### Configuration
See `configs/` directory for configuration files and templates.