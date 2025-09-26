# GDP Evaluation Framework

A comprehensive evaluation framework for Khmer language models with economic impact analysis, optimized for locally fine-tuned models.

## 🎯 Key Features

- **High-Performance Local Serving**: vLLM integration for 10x faster inference
- **Model Registry**: Centralized management for fine-tuned models
- **Khmer-Specific Metrics**: Advanced evaluation metrics for Khmer language
- **Multi-Provider Support**: vLLM, HuggingFace, Ollama, and local models
- **Economic Analysis**: Specialized evaluation for financial and economic domains
- **A/B Testing**: Compare multiple models side-by-side

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/khopilot/GDP-eval.git
cd GDP-eval

# Install dependencies
pip install -r requirements.txt

# Setup directories
mkdir -p models/store results/evaluations
```

### Basic Usage

#### 1. Register a Model
```bash
python scripts/register_model.py register \
    path/to/your/model \
    "Khmer-Economic-LLM" \
    "1.0.0" \
    "text-generation"
```

#### 2. Evaluate Model
```bash
python scripts/evaluate_khmer_model.py evaluate \
    model-v1.0 \
    data/evaluation/khmer_test_dataset.jsonl \
    --show-details
```

#### 3. Compare Models
```bash
python scripts/evaluate_khmer_model.py compare \
    model-v1 model-v2 model-v3 \
    data/evaluation/khmer_test_dataset.jsonl
```

## 📁 Project Structure

```
GDP-eval/
├── configs/               # Configuration files
│   ├── model_registry.yaml
│   └── vllm_config.yaml
├── data/                  # Datasets and tasks
│   ├── evaluation/        # Test datasets
│   ├── samples/           # Sample data
│   └── tasks/             # Task definitions
├── docs/                  # Documentation
├── examples/              # Usage examples
├── models/                # Model storage
│   ├── registry/          # Model registry DB
│   └── store/             # Model artifacts
├── results/               # Evaluation results
├── scripts/               # CLI tools
├── src/                   # Main source code
│   ├── core/              # Core framework
│   ├── evaluation/        # Evaluation pipelines
│   ├── metrics/           # Khmer metrics
│   ├── models/            # Model management
│   ├── providers/         # LLM providers
│   └── utils/             # Utilities
└── tests/                 # Test suite
```

## 🇰🇭 Khmer Language Support

### Specialized Tokenization
- Character-level with grapheme clustering
- Syllable segmentation
- Word boundary detection (ZWSP-aware)
- Code-switching detection (Khmer/English)

### Evaluation Metrics
- **Khmer BLEU**: Multi-level BLEU scores
- **Character Accuracy**: Precise character matching
- **Syllable F1**: Syllable segmentation quality
- **Word Segmentation**: Boundary detection accuracy
- **Code-Switching**: Mixed language handling

## 🔧 Advanced Features

### vLLM Provider
```python
from src.providers import VLLMProvider

provider = VLLMProvider(
    model_name="khmer-llm-7b",
    num_gpus=1,
    max_model_len=4096
)
await provider.load_model()
response = await provider.generate("តើ GDP ជាអ្វី?")
```

### Model Registry
```python
from src.models import ModelRegistry

registry = ModelRegistry()
model_id = registry.register(
    model_path="path/to/model",
    name="Khmer-FineTuned",
    version="2.0.0",
    task="text-generation",
    performance_metrics={
        "khmer_bleu": 0.85,
        "accuracy": 0.92
    }
)
```

### Custom Evaluation
```python
from src.evaluation import KhmerEvaluator, EvaluationConfig

config = EvaluationConfig(
    model_id="my-model",
    dataset_path="custom_dataset.jsonl",
    metrics_to_compute=["bleu", "character_accuracy"],
    batch_size=64
)

evaluator = KhmerEvaluator()
results = await evaluator.evaluate_model(config)
```

## 📊 Performance Benchmarks

| Model Type | Inference Speed | Memory Usage | Khmer BLEU |
|------------|----------------|--------------|------------|
| vLLM (7B)  | 150 tokens/sec | 8GB VRAM    | 0.82       |
| HF (7B)    | 15 tokens/sec  | 14GB VRAM   | 0.82       |
| GGUF (7B)  | 50 tokens/sec  | 6GB RAM     | 0.80       |

## 🛠️ Development

### Running Tests
```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Test metrics on sample text
python scripts/evaluate_khmer_model.py test
```

### Code Quality
```bash
# Format code
black src/ scripts/
ruff check --fix src/

# Type checking
mypy src/
```

## 📝 Configuration

### Model Registry (`configs/model_registry.yaml`)
- Model naming conventions
- Storage paths
- Retention policies
- Version control

### vLLM Config (`configs/vllm_config.yaml`)
- Tensor parallelism
- Quantization settings
- Memory management
- Batch processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 👤 Author

**Nicolas Delrieu** - AI Consultant
- Phone: +855 92 332 554
- Specializing in multilingual NLP and economic AI applications

## 📄 License

[Your License]

## 🆘 Support

- GitHub Issues: [https://github.com/khopilot/GDP-eval/issues](https://github.com/khopilot/GDP-eval/issues)
- Documentation: See `docs/` directory

## 🔮 Roadmap

- [ ] Text Generation Inference (TGI) provider
- [ ] MLflow integration
- [ ] FastAPI serving endpoint
- [ ] Distributed evaluation
- [ ] Real-time monitoring dashboard
- [ ] AutoGPTQ quantization support