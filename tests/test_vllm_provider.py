"""
Tests for vLLM Provider
Comprehensive testing for high-performance model serving
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers.vllm_provider import VLLMProvider, VLLMConfig, VLLM_AVAILABLE
from src.providers.base_provider import LLMResponse


class TestVLLMProvider:
    """Test suite for vLLM provider"""

    @pytest.fixture
    def mock_config(self):
        """Create mock vLLM configuration"""
        return VLLMConfig(
            model_path="test-model",
            model_name="test-model",
            gpu_memory_utilization=0.5,
            max_batch_size=32,
            tensor_parallel_size=1,
            dtype="float16"
        )

    @pytest.fixture
    def mock_vllm(self):
        """Mock vLLM module"""
        with patch('src.providers.vllm_provider.VLLM_AVAILABLE', True):
            with patch('src.providers.vllm_provider.LLM') as mock_llm:
                with patch('src.providers.vllm_provider.SamplingParams') as mock_params:
                    yield mock_llm, mock_params

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    def test_provider_initialization(self, mock_config):
        """Test provider initialization"""
        provider = VLLMProvider(mock_config)

        assert provider.provider_name == "vllm"
        assert provider.vllm_config == mock_config
        assert not provider.is_loaded
        assert provider.llm is None

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "model_path": "test-model",
            "model_name": "test-model",
            "gpu_memory_utilization": 0.8,
            "quantization": "awq"
        }

        provider = VLLMProvider(config_dict)
        assert provider.vllm_config.model_path == "test-model"
        assert provider.vllm_config.gpu_memory_utilization == 0.8
        assert provider.vllm_config.quantization == "awq"

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    @pytest.mark.asyncio
    async def test_generate_single_prompt(self, mock_config, mock_vllm):
        """Test generation with single prompt"""
        mock_llm_class, mock_sampling_params = mock_vllm

        # Create mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock output
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text", token_ids=[1, 2, 3], finish_reason="stop")]
        mock_output.prompt_token_ids = [1, 2]
        mock_llm_instance.generate.return_value = [mock_output]

        # Mock model config
        mock_model_config = MagicMock()
        mock_model_config.dtype = "float16"
        mock_model_config.max_model_len = 4096
        mock_model_config.get_vocab_size.return_value = 50000
        mock_llm_instance.llm_engine.model_config = mock_model_config

        provider = VLLMProvider(mock_config)
        provider.load_model()

        response = await provider.generate("Test prompt", max_tokens=100)

        assert isinstance(response, LLMResponse)
        assert response.text == "Generated text"
        assert response.provider == "vllm"
        assert response.error is None
        assert response.tokens_used == 5  # 2 prompt + 3 completion

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    @pytest.mark.asyncio
    async def test_generate_batch(self, mock_config, mock_vllm):
        """Test batch generation"""
        mock_llm_class, mock_sampling_params = mock_vllm

        # Create mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock outputs for batch
        mock_outputs = []
        for i in range(3):
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock(
                text=f"Generated text {i}",
                token_ids=[1, 2, 3],
                finish_reason="stop"
            )]
            mock_output.prompt_token_ids = [1, 2]
            mock_outputs.append(mock_output)

        mock_llm_instance.generate.return_value = mock_outputs

        # Mock model config
        mock_model_config = MagicMock()
        mock_model_config.dtype = "float16"
        mock_model_config.max_model_len = 4096
        mock_model_config.get_vocab_size.return_value = 50000
        mock_llm_instance.llm_engine.model_config = mock_model_config

        provider = VLLMProvider(mock_config)
        provider.load_model()

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await provider.generate(prompts, max_tokens=100)

        assert isinstance(responses, list)
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.text == f"Generated text {i}"
            assert response.metadata["batch_size"] == 3

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    def test_model_info(self, mock_config, mock_vllm):
        """Test getting model information"""
        mock_llm_class, _ = mock_vllm

        # Create mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock model config
        mock_model_config = MagicMock()
        mock_model_config.dtype = "float16"
        mock_model_config.max_model_len = 4096
        mock_model_config.get_vocab_size.return_value = 50000
        mock_llm_instance.llm_engine.model_config = mock_model_config

        provider = VLLMProvider(mock_config)
        provider.load_model()

        info = provider.get_model_info()

        assert info["engine"] == "vllm"
        assert info["is_loaded"] == True
        assert info["gpu_memory_utilization"] == 0.5
        assert info["tensor_parallel_size"] == 1
        assert info["max_model_len"] == 4096
        assert info["vocab_size"] == 50000

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    def test_unload_model(self, mock_config, mock_vllm):
        """Test model unloading"""
        mock_llm_class, _ = mock_vllm

        # Create mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock model config
        mock_model_config = MagicMock()
        mock_llm_instance.llm_engine.model_config = mock_model_config

        provider = VLLMProvider(mock_config)
        provider.load_model()

        assert provider.is_loaded
        assert provider.llm is not None

        provider.unload_model()

        assert not provider.is_loaded
        assert provider.llm is None

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    def test_benchmark_batch_sizes(self, mock_config, mock_vllm):
        """Test batch size benchmarking"""
        mock_llm_class, mock_sampling_params = mock_vllm

        # Create mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock model config
        mock_model_config = MagicMock()
        mock_llm_instance.llm_engine.model_config = mock_model_config

        def generate_mock_outputs(prompts, _):
            outputs = []
            for _ in prompts:
                mock_output = MagicMock()
                mock_output.outputs = [MagicMock(
                    text="Test",
                    token_ids=[1, 2],
                    finish_reason="stop"
                )]
                mock_output.prompt_token_ids = [1]
                outputs.append(mock_output)
            return outputs

        mock_llm_instance.generate.side_effect = generate_mock_outputs

        provider = VLLMProvider(mock_config)
        provider.load_model()

        results = provider.benchmark_batch_sizes(
            prompt="Test",
            batch_sizes=[1, 2, 4],
            max_tokens=10
        )

        assert "optimal_batch_size" in results
        assert "max_throughput_req_per_s" in results
        assert "batch_1" in results
        assert "batch_2" in results
        assert "batch_4" in results

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config, mock_vllm):
        """Test error handling in generation"""
        mock_llm_class, _ = mock_vllm

        # Create mock LLM instance that raises error
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = Exception("Test error")
        mock_llm_class.return_value = mock_llm_instance

        # Mock model config
        mock_model_config = MagicMock()
        mock_llm_instance.llm_engine.model_config = mock_model_config

        provider = VLLMProvider(mock_config)
        provider.load_model()

        response = await provider.generate("Test prompt")

        assert response.error == "Test error"
        assert response.text == ""

    @pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not installed")
    def test_from_pretrained(self, mock_vllm):
        """Test creating provider from pretrained model"""
        mock_llm_class, _ = mock_vllm

        # Create mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        # Mock model config
        mock_model_config = MagicMock()
        mock_llm_instance.llm_engine.model_config = mock_model_config

        provider = VLLMProvider.from_pretrained(
            "test-model",
            gpu_memory_utilization=0.7
        )

        assert provider.vllm_config.model_path == "test-model"
        assert provider.vllm_config.gpu_memory_utilization == 0.7
        assert provider.is_loaded


class TestVLLMIntegration:
    """Integration tests for vLLM provider (requires GPU and vLLM)"""

    @pytest.mark.skip(reason="Requires GPU and actual model")
    @pytest.mark.asyncio
    async def test_real_model_inference(self):
        """Test with real model (manual test)"""
        config = VLLMConfig(
            model_path="facebook/opt-125m",  # Small model for testing
            model_name="opt-125m",
            gpu_memory_utilization=0.3,
            max_batch_size=8
        )

        provider = VLLMProvider(config)
        provider.load_model()

        # Test single generation
        response = await provider.generate(
            "The capital of France is",
            max_tokens=10
        )

        assert response.text
        assert not response.error
        assert response.tokens_used > 0

        # Test batch generation
        prompts = [
            "What is GDP?",
            "តើ GDP ជាអ្វី?",
            "Explain economics."
        ]

        responses = await provider.generate(prompts, max_tokens=50)
        assert len(responses) == 3

        # Cleanup
        provider.unload_model()

    @pytest.mark.skip(reason="Requires GPU")
    def test_memory_usage(self):
        """Test memory usage tracking"""
        config = VLLMConfig(
            model_path="facebook/opt-125m",
            model_name="opt-125m",
            gpu_memory_utilization=0.3
        )

        provider = VLLMProvider(config)

        # Check memory before loading
        memory_before = provider.get_memory_usage()

        # Load model
        provider.load_model()

        # Check memory after loading
        memory_after = provider.get_memory_usage()

        # Verify GPU memory increased
        if "gpu_0_used_gb" in memory_after:
            assert memory_after["gpu_0_used_gb"] > memory_before.get("gpu_0_used_gb", 0)

        # Unload and check memory released
        provider.unload_model()
        memory_final = provider.get_memory_usage()

        if "gpu_0_used_gb" in memory_final:
            assert memory_final["gpu_0_used_gb"] < memory_after["gpu_0_used_gb"]