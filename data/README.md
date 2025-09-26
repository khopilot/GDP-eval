# Data Directory

This directory contains datasets, task definitions, and evaluation data.

## Structure

- `evaluation/` - Evaluation datasets for model testing
  - `khmer_test_dataset.jsonl` - Khmer economic Q&A pairs

- `tasks/` - Task definitions for evaluation
  - `sample_tasks.json` - Example task configurations

- `samples/` - Sample data and examples
  - `task-json-example.json` - Task format example

## Data Formats

### Evaluation Dataset (JSONL)
```json
{"prompt": "Question in Khmer or English", "reference": "Expected answer"}
```

### Task Definition (JSON)
```json
{
  "task_id": "unique-id",
  "type": "generation|classification|translation",
  "prompt": "Task prompt",
  "context": "Optional context",
  "expected": "Expected output"
}
```

## Adding New Datasets

1. Place evaluation datasets in `evaluation/`
2. Use JSONL format for streaming large datasets
3. Include both Khmer and English examples for bilingual evaluation
4. Document dataset sources and preprocessing steps

## Dataset Guidelines

- Always include reference/ground truth for evaluation
- Balance dataset between different task types
- Include edge cases and challenging examples
- Document any data cleaning or preprocessing