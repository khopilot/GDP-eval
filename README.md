# GDP Evaluation Framework

A comprehensive Python framework for GDP (Gross Domestic Product) evaluation and analysis with specialized support for Khmer language processing.

## Features

- **Economic Analysis**: Advanced GDP data processing and validation
- **Multi-language Support**: Full Khmer language support for economic reports
- **Automated Grading**: Intelligent evaluation and grading system
- **Data Validation**: Robust data quality checks and normalization
- **Extensible Architecture**: Modular design for easy customization

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GDP-eval
```

2. Set up the environment:
```bash
.claude/commands.sh setup
```

### Usage

#### Run Economic Analysis
```bash
python economic-analyzer.py --country KHM --year 2024
```

#### Evaluate GDP Data
```bash
python evaluation-script.py --input data.json
```

#### Grade Submissions
```bash
python grading-system.py --batch submissions/
```

#### Test Khmer Utilities
```bash
python khmer-utils.py --test
```

## Project Structure

```
GDP-eval/
├── economic-analyzer.py      # Core economic analysis engine
├── evaluation-script.py       # Main evaluation orchestrator
├── gdpval-framework.py       # GDP validation framework
├── grading-system.py         # Automated grading system
├── khmer-utils.py           # Khmer language utilities
├── task-json-example.json   # Task configuration examples
├── CLAUDE.md                # Claude Code configuration
└── .claude/                 # Project-specific Claude settings
    ├── settings.local.json  # Local configuration
    ├── agent-tasks.json     # Sub-agent task definitions
    └── commands.sh          # Quick command scripts
```

## Claude Code Integration

This project is optimized for Claude Code with:

- **Automated Formatting**: Black and Ruff integration
- **Type Checking**: MyPy configuration
- **Sub-Agents**: Specialized agents for data curation, Khmer NLP, and model evaluation
- **Parallel Execution**: Run multiple analyses concurrently

### Available Commands

```bash
# Format code
.claude/commands.sh format

# Run tests
.claude/commands.sh test

# Check code quality
.claude/commands.sh check

# Backup project
.claude/commands.sh backup
```

### Sub-Agent Tasks

Launch specialized agents for complex tasks:

- **data-curator-annotator**: Data preparation and validation
- **khmer-nlp-specialist**: Khmer text processing
- **model-evaluation-qa**: Model accuracy assessment
- **llm-finetuning-engineer**: Model optimization

## Development

### Code Style

This project follows:
- PEP 8 with Black formatting
- Type hints for all functions
- Google-style docstrings
- Comprehensive error handling

### Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Contributing

Please read CLAUDE.md for detailed development guidelines and coding standards.

## License

[Your License Here]

## Support

For issues or questions, please open an issue on the repository.