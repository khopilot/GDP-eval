"""
Integration tests for evaluation pipeline
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.evaluation.khmer_evaluator import KhmerEvaluator, EvaluationConfig, EvaluationResult
from src.metrics.khmer_metrics import KhmerMetrics
from src.models.model_registry import ModelRegistry


class TestEvaluationPipeline:
    """Test complete evaluation pipeline"""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return KhmerEvaluator()

    @pytest.fixture
    def eval_config(self):
        """Create evaluation configuration"""
        return EvaluationConfig(
            model_id="test-model-v1",
            dataset_path="data/evaluation/khmer_test_dataset.jsonl",
            metrics_to_compute=["bleu", "character_accuracy"],
            batch_size=2,
            max_samples=5
        )

    @pytest.fixture
    def mock_provider(self):
        """Create mock vLLM provider"""
        provider = AsyncMock()
        provider.load_model = AsyncMock()
        provider.generate = AsyncMock(return_value=Mock(
            content="Generated text",
            success=True
        ))
        provider.batch_generate = AsyncMock(return_value=[
            Mock(content="Generated 1", success=True),
            Mock(content="Generated 2", success=True)
        ])
        return provider

    @pytest.mark.asyncio
    async def test_evaluation_with_mock_provider(self, evaluator, eval_config, mock_provider):
        """Test evaluation with mocked provider"""
        # Mock the provider loading
        with patch.object(evaluator, 'provider', mock_provider):
            with patch.object(evaluator, '_load_model', AsyncMock()):
                # Mock dataset loading
                mock_dataset = {
                    'prompts': ["Test prompt 1", "Test prompt 2"],
                    'references': ["Reference 1", "Reference 2"]
                }
                with patch.object(evaluator, '_load_dataset', return_value=mock_dataset):
                    # Run evaluation
                    result = await evaluator.evaluate_model(eval_config)

                    assert isinstance(result, EvaluationResult)
                    assert result.model_id == eval_config.model_id
                    assert len(result.predictions) == len(mock_dataset['prompts'])
                    assert 'bleu' in result.metrics

    def test_load_jsonl_dataset(self, evaluator, tmp_path):
        """Test loading JSONL dataset"""
        # Create test JSONL file
        test_file = tmp_path / "test.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('{"prompt": "Test 1", "reference": "Ref 1"}\n')
            f.write('{"prompt": "Test 2", "reference": "Ref 2"}\n')

        dataset = evaluator._load_dataset(str(test_file))

        assert 'prompts' in dataset
        assert 'references' in dataset
        assert len(dataset['prompts']) == 2
        assert dataset['prompts'][0] == "Test 1"

    def test_load_json_dataset(self, evaluator, tmp_path):
        """Test loading JSON dataset"""
        test_file = tmp_path / "test.json"
        data = [
            {"input": "Test 1", "output": "Ref 1"},
            {"question": "Test 2", "answer": "Ref 2"}
        ]
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        dataset = evaluator._load_dataset(str(test_file))

        assert len(dataset['prompts']) == 2
        assert len(dataset['references']) == 2

    def test_compute_metrics(self, evaluator):
        """Test metrics computation"""
        predictions = ["ធនាគារជាតិ", "កម្ពុជា"]
        references = ["ធនាគារជាតិ", "កម្ពុជា"]

        results = evaluator._compute_metrics(
            predictions, references,
            metrics_to_compute=["bleu", "character_accuracy"]
        )

        assert 'bleu' in results
        assert 'character_accuracy' in results
        assert results['character_accuracy'].score == 1.0  # Perfect match

    def test_compute_all_metrics(self, evaluator):
        """Test computing all metrics"""
        predictions = ["ធនាគារ", "GDP"]
        references = ["ធនាគារជាតិ", "GDP កម្ពុជា"]

        results = evaluator._compute_metrics(
            predictions, references,
            metrics_to_compute=["all"]
        )

        # Should have all metric types
        assert len(results) > 5
        assert 'bleu_syllable' in results
        assert 'character_accuracy' in results
        assert 'syllable_f1' in results

    @pytest.mark.asyncio
    async def test_compare_models(self, evaluator, tmp_path):
        """Test model comparison"""
        # Mock evaluation results
        async def mock_evaluate(config):
            return EvaluationResult(
                model_id=config.model_id,
                dataset=config.dataset_path,
                metrics={'bleu': 0.75, 'accuracy': 0.85},
                detailed_metrics={},
                predictions=["pred1"],
                references=["ref1"],
                prompts=["prompt1"],
                evaluation_time=10.0,
                timestamp="2024-01-01",
                config=config
            )

        with patch.object(evaluator, 'evaluate_model', mock_evaluate):
            df = await evaluator.compare_models(
                ["model1", "model2"],
                "test_dataset.jsonl"
            )

            assert len(df) == 2
            assert "model1" in df.index
            assert "model2" in df.index
            assert "overall_score" in df.columns

    def test_create_leaderboard(self, evaluator, tmp_path):
        """Test leaderboard creation"""
        # Create test evaluation results
        eval_dir = tmp_path / "evaluations"
        eval_dir.mkdir()

        result1 = {
            "model_id": "model1",
            "dataset": "test.jsonl",
            "metrics": {"bleu": 0.8, "accuracy": 0.9},
            "num_samples": 100
        }

        result2 = {
            "model_id": "model2",
            "dataset": "test.jsonl",
            "metrics": {"bleu": 0.75, "accuracy": 0.85},
            "num_samples": 100
        }

        with open(eval_dir / "model1_evaluation.json", 'w') as f:
            json.dump(result1, f)

        with open(eval_dir / "model2_evaluation.json", 'w') as f:
            json.dump(result2, f)

        df = evaluator.create_leaderboard(str(eval_dir))

        assert len(df) == 2
        assert "rank" in df.columns
        assert df.iloc[0]["model_id"] == "model1"  # Higher score should rank first

    def test_evaluation_result_save(self, tmp_path):
        """Test saving evaluation results"""
        config = EvaluationConfig(
            model_id="test-model",
            dataset_path="test.jsonl",
            save_predictions=True
        )

        result = EvaluationResult(
            model_id="test-model",
            dataset="test.jsonl",
            metrics={"bleu": 0.75},
            detailed_metrics={},
            predictions=["pred1", "pred2"],
            references=["ref1", "ref2"],
            prompts=["prompt1", "prompt2"],
            evaluation_time=10.0,
            timestamp="2024-01-01",
            config=config
        )

        output_file = tmp_path / "result.json"
        result.save(str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert data["model_id"] == "test-model"
            assert data["metrics"]["bleu"] == 0.75

        # Check predictions file
        pred_file = output_file.with_suffix('.predictions.jsonl')
        assert pred_file.exists()

    @pytest.mark.asyncio
    async def test_error_handling(self, evaluator, eval_config):
        """Test error handling in evaluation"""
        # Test with non-existent model
        with patch.object(evaluator.registry, 'get_model', return_value=None):
            with pytest.raises(ValueError, match="Model not found"):
                await evaluator.evaluate_model(eval_config)

    def test_metric_selection(self, evaluator):
        """Test specific metric selection"""
        predictions = ["text"]
        references = ["text"]

        # Test unknown metric
        results = evaluator._compute_metrics(
            predictions, references,
            metrics_to_compute=["unknown_metric"]
        )

        assert len(results) == 0  # Unknown metric should be skipped

        # Test valid metrics
        results = evaluator._compute_metrics(
            predictions, references,
            metrics_to_compute=["bleu", "edit_distance"]
        )

        assert "bleu" in results
        assert "edit_distance" in results