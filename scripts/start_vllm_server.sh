#!/bin/bash

# vLLM Server Startup Script
# High-performance model serving for fine-tuned Khmer models

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
MODEL_PATH="models/khmer-llama-7b"
PORT=8000
HOST="0.0.0.0"
GPU_MEMORY=0.9
TENSOR_PARALLEL=1
MAX_BATCH_SIZE=256
CONFIG_FILE="configs/vllm_config.yaml"
MODE="server"  # server or offline

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH          Path to model (default: models/khmer-llama-7b)"
            echo "  --port PORT          Server port (default: 8000)"
            echo "  --host HOST          Server host (default: 0.0.0.0)"
            echo "  --gpu-memory FRAC    GPU memory fraction (default: 0.9)"
            echo "  --tensor-parallel N  Number of GPUs for tensor parallelism (default: 1)"
            echo "  --config FILE        Config file path (default: configs/vllm_config.yaml)"
            echo "  --mode MODE         'server' or 'offline' (default: server)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Start server with default settings"
            echo "  $0"
            echo ""
            echo "  # Use specific model with 2 GPUs"
            echo "  $0 --model models/my-model --tensor-parallel 2"
            echo ""
            echo "  # Run in offline mode for testing"
            echo "  $0 --mode offline"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for vLLM installation
log_info "Checking vLLM installation..."
if ! python -c "import vllm" 2>/dev/null; then
    log_error "vLLM is not installed"
    log_info "Install with: pip install vllm"
    exit 1
fi

# Check for CUDA
log_info "Checking CUDA availability..."
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log_warn "CUDA is not available. Performance will be limited."
    TENSOR_PARALLEL=1
else
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    log_info "Found $GPU_COUNT GPU(s)"

    # Adjust tensor parallel size if needed
    if [ "$TENSOR_PARALLEL" -gt "$GPU_COUNT" ]; then
        log_warn "Requested $TENSOR_PARALLEL GPUs but only $GPU_COUNT available"
        TENSOR_PARALLEL=$GPU_COUNT
    fi
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    log_warn "Model not found at $MODEL_PATH"
    log_info "Attempting to download from HuggingFace..."
    # Model will be downloaded automatically by vLLM
fi

# Create logs directory
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TENSOR_PARALLEL-1)))
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

log_info "Starting vLLM with:"
log_info "  Model: $MODEL_PATH"
log_info "  Mode: $MODE"
log_info "  Tensor Parallel: $TENSOR_PARALLEL"
log_info "  GPU Memory: $GPU_MEMORY"
log_info "  CUDA Devices: $CUDA_VISIBLE_DEVICES"

if [ "$MODE" = "server" ]; then
    # Start vLLM API server
    log_info "Starting vLLM API server on $HOST:$PORT"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --max-num-seqs "$MAX_BATCH_SIZE" \
        --trust-remote-code \
        --dtype auto \
        --seed 42 \
        2>&1 | tee logs/vllm_server_$(date +%Y%m%d_%H%M%S).log

elif [ "$MODE" = "offline" ]; then
    # Run offline test script
    log_info "Running offline vLLM test..."

    python - << EOF
import sys
sys.path.insert(0, '.')

from src.providers.vllm_provider import VLLMProvider, VLLMConfig
import asyncio

async def test_vllm():
    # Create configuration
    config = VLLMConfig(
        model_path="$MODEL_PATH",
        model_name="$(basename $MODEL_PATH)",
        gpu_memory_utilization=$GPU_MEMORY,
        tensor_parallel_size=$TENSOR_PARALLEL,
        max_batch_size=$MAX_BATCH_SIZE
    )

    # Initialize provider
    provider = VLLMProvider(config)
    provider.load_model()

    # Test prompts
    test_prompts = [
        "What is the GDP of Cambodia?",
        "តើ GDP របស់កម្ពុជាគឺជាអ្វី?",
        "Explain mobile banking impact on economy.",
        "ពន្យល់ពីផលប៉ះពាល់នៃធនាគារតាមទូរស័ព្ទលើសេដ្ឋកិច្ច។"
    ]

    print("\n" + "="*50)
    print("Testing vLLM Provider")
    print("="*50)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        response = await provider.generate(prompt, max_tokens=100)
        if response.error:
            print(f"Error: {response.error}")
        else:
            print(f"Response: {response.text[:200]}...")
            print(f"Latency: {response.latency_ms:.2f}ms")
            print(f"Tokens: {response.tokens_used}")

    # Benchmark
    print("\n" + "="*50)
    print("Running Benchmark")
    print("="*50)

    results = provider.benchmark(
        prompts=["Generate a financial report summary."] * 3,
        max_tokens=100,
        num_iterations=5
    )

    print(f"Mean latency: {results['latency_ms']['mean']:.2f}ms")
    print(f"P95 latency: {results['latency_ms']['p95']:.2f}ms")
    print(f"Throughput: {results['throughput']['tokens_per_second_mean']:.2f} tokens/s")

    # Test batch processing
    print("\n" + "="*50)
    print("Testing Batch Processing")
    print("="*50)

    batch_results = provider.benchmark_batch_sizes(
        prompt="Analyze economic data.",
        batch_sizes=[1, 4, 8, 16],
        max_tokens=50
    )

    print(f"Optimal batch size: {batch_results['optimal_batch_size']}")
    print(f"Max throughput: {batch_results['max_throughput_req_per_s']:.2f} req/s")

    # Cleanup
    provider.unload_model()
    print("\nTest complete!")

# Run test
asyncio.run(test_vllm())
EOF

else
    log_error "Invalid mode: $MODE (use 'server' or 'offline')"
    exit 1
fi

log_info "vLLM shutdown complete"