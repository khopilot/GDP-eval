#!/bin/bash
# Claude Code Quick Commands for GDP-eval Project

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Command: setup-python
setup_python() {
    log_info "Setting up Python environment..."

    # Check Python version
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ $(echo "$python_version >= 3.9" | bc -l) -eq 1 ]]; then
        log_info "Python $python_version detected ✓"
    else
        log_error "Python 3.9+ required, found $python_version"
        return 1
    fi

    # Create virtual environment if not exists
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate and install dependencies
    source venv/bin/activate

    # Install development tools
    pip install --upgrade pip
    pip install black ruff mypy pytest pytest-cov
    pip install pandas numpy

    log_info "Python environment ready!"
}

# Command: run-evaluation
run_evaluation() {
    log_info "Running GDP evaluation..."
    python evaluation-script.py "$@"
}

# Command: run-analysis
run_analysis() {
    log_info "Running economic analysis..."
    python economic-analyzer.py "$@"
}

# Command: run-grading
run_grading() {
    log_info "Running grading system..."
    python grading-system.py "$@"
}

# Command: test-khmer
test_khmer() {
    log_info "Testing Khmer utilities..."
    python -c "import khmer_utils; print('Khmer utils loaded successfully')"
    python khmer-utils.py --test
}

# Command: format-code
format_code() {
    log_info "Formatting Python code..."
    black *.py
    ruff check --fix *.py
}

# Command: run-tests
run_tests() {
    log_info "Running tests..."
    if [ -d "tests" ]; then
        python -m pytest tests/ -v
    else
        log_warn "No tests directory found"
    fi
}

# Command: check-quality
check_quality() {
    log_info "Checking code quality..."

    # Format check
    black --check *.py

    # Linting
    ruff check *.py

    # Type checking
    if [ -f "mypy.ini" ] || [ -f "pyproject.toml" ]; then
        mypy *.py
    else
        log_warn "No mypy configuration found"
    fi
}

# Command: parallel-analyze
parallel_analyze() {
    log_info "Running parallel analysis with sub-agents..."

    # This would trigger Claude Code sub-agents
    echo "Use /agent/parallel command in Claude Code to run:"
    echo "1. data-curator-annotator: Validate all datasets"
    echo "2. model-evaluation-qa: Evaluate analysis accuracy"
    echo "3. khmer-nlp-specialist: Process Khmer text"
}

# Command: monitor-agents
monitor_agents() {
    log_info "Monitoring sub-agent execution..."
    if [ -f "$HOME/.claude/scripts/monitor-agents.py" ]; then
        python "$HOME/.claude/scripts/monitor-agents.py"
    else
        log_warn "Monitor script not found in ~/.claude/scripts/"
    fi
}

# Command: backup-project
backup_project() {
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="backups/gdp-eval-$timestamp"

    log_info "Creating backup in $backup_dir..."
    mkdir -p "$backup_dir"

    # Backup Python files and data
    cp *.py "$backup_dir/" 2>/dev/null
    cp *.json "$backup_dir/" 2>/dev/null
    cp *.md "$backup_dir/" 2>/dev/null

    # Backup Claude configuration
    cp -r .claude "$backup_dir/" 2>/dev/null

    log_info "Backup created successfully!"
}

# Command: clean-project
clean_project() {
    log_info "Cleaning project..."

    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete 2>/dev/null
    find . -type f -name "*.pyo" -delete 2>/dev/null

    # Remove test artifacts
    rm -rf .pytest_cache 2>/dev/null
    rm -rf .mypy_cache 2>/dev/null
    rm -rf .ruff_cache 2>/dev/null
    rm -f .coverage 2>/dev/null

    log_info "Project cleaned!"
}

# Main command handler
case "$1" in
    setup)
        setup_python
        ;;
    evaluate)
        shift
        run_evaluation "$@"
        ;;
    analyze)
        shift
        run_analysis "$@"
        ;;
    grade)
        shift
        run_grading "$@"
        ;;
    test-khmer)
        test_khmer
        ;;
    format)
        format_code
        ;;
    test)
        run_tests
        ;;
    check)
        check_quality
        ;;
    parallel)
        parallel_analyze
        ;;
    monitor)
        monitor_agents
        ;;
    backup)
        backup_project
        ;;
    clean)
        clean_project
        ;;
    *)
        echo "GDP-eval Project Commands:"
        echo "  setup       - Set up Python environment"
        echo "  evaluate    - Run evaluation script"
        echo "  analyze     - Run economic analyzer"
        echo "  grade       - Run grading system"
        echo "  test-khmer  - Test Khmer utilities"
        echo "  format      - Format code with Black"
        echo "  test        - Run pytest tests"
        echo "  check       - Check code quality"
        echo "  parallel    - Info about parallel analysis"
        echo "  monitor     - Monitor sub-agents"
        echo "  backup      - Backup project"
        echo "  clean       - Clean project files"
        ;;
esac