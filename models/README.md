# Models Directory

This directory contains the model registry and storage for fine-tuned models.

## Structure

- `registry/` - Model registry database and metadata
- `store/` - Physical model storage
  - `checkpoints/` - Full model checkpoints
  - `adapters/` - LoRA/QLoRA adapters
  - `quantized/` - Quantized models (GPTQ, AWQ, GGUF)
  - `exports/` - Exported models (ONNX, etc.)

## Usage

### Register a model
```bash
python scripts/register_model.py register \
    models/store/checkpoints/my-model \
    "My Model" \
    "1.0.0" \
    "text-generation"
```

### List registered models
```bash
python scripts/register_model.py list
```

### Get model details
```bash
python scripts/register_model.py get model-v1.0
```

## Storage Management

Models are automatically organized by type and version. Large model files are excluded from git tracking.

## Model Naming Convention

- Base models: `base-{name}-v{version}`
- Fine-tuned: `ft-{name}-v{version}`
- LoRA adapters: `lora-{name}-v{version}`
- Quantized: `quant-{name}-{method}-v{version}`