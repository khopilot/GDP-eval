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
└── DEVELOPMENT.md           # Development guidelines
```

## Development Features

This project includes:

- **Automated Formatting**: Black and Ruff integration
- **Type Checking**: MyPy configuration
- **Task Automation**: Specialized task runners for data curation, Khmer NLP, and model evaluation
- **Parallel Execution**: Run multiple analyses concurrently

### Quick Commands

```bash
# Setup environment
./setup.sh

# Run demo
python demo.py

# Format code
black . && ruff check --fix .

# Run tests
python -m pytest tests/ -v
```

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

## Author

**Nicolas Delrieu** - AI Consultant
Specializing in AI/ML solutions and multilingual NLP systems

## Contributing

Please read the development guidelines for detailed coding standards and best practices.

## License

[Your License Here]

## Support

For issues or questions, please open an issue on the repository.