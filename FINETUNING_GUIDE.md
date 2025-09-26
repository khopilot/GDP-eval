# Fine-Tuning Evaluation Guide

## Quick Start: Testing Your Fine-Tuned Model

### 1. Choose Your Serving Method

The framework supports multiple ways to serve your fine-tuned model:

#### Option A: Ollama (Simplest)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Load your model
ollama create your-model-name -f Modelfile

# Run Ollama server
ollama serve
```

#### Option B: vLLM (Fastest)
```bash
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --port 8000
```

#### Option C: HuggingFace (Most Flexible)
```python
# Your model should be on HuggingFace Hub or local path
model_name = "your-username/your-model"
# or
model_path = "/path/to/your/model"
```

### 2. Run Comparison

Edit `compare_models.py` to add your model:

```python
model_configs = [
    {
        "name": "grok-baseline",
        "type": "grok",
        "api_key": "your-api-key",
        "model": "grok-3",
        "timeout": 120
    },
    {
        "name": "your-finetuned-model",
        "type": "ollama",  # or "vllm" or "huggingface"
        "model": "your-model-name",
        "base_url": "http://localhost:11434"  # for ollama
    }
]
```

Run the comparison:
```bash
python compare_models.py
```

### 3. Interpret Results

The comparison will show:

```
| Model              | Success Rate | Avg Latency | Tokens | Cost | Efficiency |
|-------------------|--------------|-------------|--------|------|------------|
| grok-baseline     |      40.0%   |    45000ms  |  2371  | $0.0044 |    16.5% |
| your-model        |      ??%     |    ????ms   |  ????  | $0.00   |    ??%   |

IMPROVEMENT: your-model vs grok-baseline
Success Rate: +XX%
Latency: -XXXXms
Efficiency: +XX%
```

## What to Look For

### Before Fine-Tuning (Baseline)
- Success Rate: ~40% (many timeouts)
- Latency: 45-60 seconds
- Khmer Support: Limited
- Professional Context: Generic

### After Fine-Tuning (Target)
- Success Rate: 80%+ (reliable completion)
- Latency: <30 seconds
- Khmer Support: Native understanding
- Professional Context: Cambodia-specific

## Test Tasks

The framework tests 5 real Cambodia professional scenarios:

1. **Finance**: SME loan assessment in Phnom Penh
2. **Agriculture**: Rice farming advice for Battambang
3. **Tourism**: Tour package for Angkor Wat
4. **Manufacturing**: Garment quality control
5. **Healthcare**: Rural clinic triage

## Key Metrics

### Pragmatic Success Indicators
- **Task Completion**: Did it provide a usable answer?
- **Response Time**: Fast enough for real use?
- **Language Handling**: Can work with Khmer/English?
- **Professional Relevance**: Understands local context?

### Economic Impact
- Time saved per task (hours)
- Cost savings per year (USD)
- Productivity gain (%)

## Troubleshooting

### If tasks timeout:
- Increase timeout in provider configuration
- Reduce max_tokens
- Simplify prompts

### If Khmer text issues:
- Ensure model supports Unicode/UTF-8
- Check tokenizer configuration
- Verify ZWSP handling

### If low quality scores:
- This is normal for baseline
- Focus on completion rate first
- Quality improves with fine-tuning

## Example: Testing with Ollama

```bash
# 1. Prepare your model
ollama pull llama2  # or load your custom model

# 2. Edit compare_models.py
# Change model_configs to include your model

# 3. Run comparison
python compare_models.py

# 4. Check results
cat results/comparison/comparison_*.json
```

## Support

- Test with 3 tasks first (`task_limit=3`)
- Use consistent temperature (0.7)
- Keep max_tokens at 1024
- Run multiple times for consistency

Your fine-tuned model should show clear improvements in:
1. Understanding Khmer context
2. Faster response times
3. Higher success rates
4. More relevant professional answers

Good luck with your fine-tuning!