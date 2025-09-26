# GDP Evaluation Framework - Claude Configuration

## Project Overview
GDP (Gross Domestic Product) evaluation framework with Khmer language support.
Python-based analysis and grading system for economic data processing.

## Project Structure
- `economic-analyzer.py` - Core economic analysis engine
- `evaluation-script.py` - Main evaluation orchestrator
- `gdpval-framework.py` - GDP validation framework
- `grading-system.py` - Automated grading and assessment
- `khmer-utils.py` - Khmer language utilities and processing
- `task-json-example.json` - Task configuration examples

## Development Guidelines

### Python Standards
- Python 3.9+ required
- Type hints for all functions
- Docstrings following Google style
- PEP 8 compliance with Black formatting
- Use pytest for testing

### Code Style
```python
def analyze_gdp_data(
    data: pd.DataFrame,
    country: str,
    year: int,
    *,
    validate: bool = True
) -> Dict[str, Any]:
    """Analyze GDP data for a specific country and year.

    Args:
        data: Input DataFrame containing GDP metrics
        country: ISO 3166-1 alpha-3 country code
        year: Target year for analysis
        validate: Whether to validate input data

    Returns:
        Dictionary containing analysis results

    Raises:
        ValueError: If country or year is invalid
        DataError: If data validation fails
    """
```

### Error Handling
- Use custom exceptions for domain errors
- Log errors with proper context
- Return Result types for recoverable errors
- Never silently catch exceptions

### Data Processing
- Validate all inputs before processing
- Use pandas for tabular data
- NumPy for numerical computations
- Handle missing data explicitly

### Khmer Language Support
- UTF-8 encoding for all Khmer text
- Use `khmer-utils.py` for text processing
- Test with real Khmer data samples
- Maintain bidirectional text compatibility

## Testing Strategy
- Unit tests for core functions
- Integration tests for workflows
- Data validation tests
- Khmer text handling tests
- Performance benchmarks for large datasets

## Quick Commands
```bash
# Run evaluation
python evaluation-script.py --input data.json

# Analyze economic data
python economic-analyzer.py --country KHM --year 2024

# Grade submissions
python grading-system.py --batch submissions/

# Test Khmer utilities
python -m pytest tests/test_khmer_utils.py
```

## Performance Considerations
- Batch process large datasets
- Use vectorized operations with NumPy/pandas
- Cache intermediate results
- Profile memory usage for large analyses

## Security Notes
- Validate all JSON inputs
- Sanitize file paths
- No arbitrary code execution
- Secure API credentials in environment variables

## Dependencies
- pandas >= 2.0
- numpy >= 1.24
- pytest >= 7.0
- black >= 23.0
- mypy >= 1.0

## CI/CD Pipeline
Refer to `github-workflow.txt` for GitHub Actions configuration.

## Sub-Agent Tasks
- **data-curator**: Prepare and validate economic datasets
- **llm-finetuning**: Fine-tune models for Khmer economic terminology
- **model-evaluation**: Evaluate GDP prediction accuracy
- **khmer-nlp**: Process Khmer economic reports