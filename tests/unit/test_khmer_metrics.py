"""
Unit tests for Khmer metrics
"""

import pytest
import numpy as np

from src.metrics.khmer_metrics import KhmerMetrics, KhmerMetricResult


class TestKhmerMetrics:
    """Test Khmer metrics functionality"""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance"""
        return KhmerMetrics()

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing"""
        return {
            'predictions': [
                "ធនាគារជាតិកម្ពុជា",
                "GDP កម្ពុជា"
            ],
            'references': [
                "ធនាគារជាតិនៃកម្ពុជា",
                "GDP របស់កម្ពុជា"
            ]
        }

    def test_calculate_khmer_bleu(self, metrics, sample_texts):
        """Test BLEU score calculation"""
        result = metrics.calculate_khmer_bleu(
            sample_texts['predictions'],
            sample_texts['references'],
            tokenization='syllable'
        )

        assert isinstance(result, KhmerMetricResult)
        assert 0 <= result.score <= 1
        assert result.metric_name == "khmer_bleu_syllable"
        assert 'tokenization' in result.details

    def test_bleu_different_tokenizations(self, metrics, sample_texts):
        """Test BLEU with different tokenization methods"""
        tokenizations = ['char', 'syllable', 'word']
        results = {}

        for tokenization in tokenizations:
            result = metrics.calculate_khmer_bleu(
                sample_texts['predictions'],
                sample_texts['references'],
                tokenization=tokenization
            )
            results[tokenization] = result.score

        # Different tokenizations should give different scores
        assert len(set(results.values())) > 1

    def test_character_accuracy(self, metrics, sample_texts):
        """Test character accuracy calculation"""
        result = metrics.calculate_character_accuracy(
            sample_texts['predictions'],
            sample_texts['references']
        )

        assert isinstance(result, KhmerMetricResult)
        assert 0 <= result.score <= 1
        assert result.metric_name == "character_accuracy"

    def test_character_accuracy_with_normalization(self, metrics):
        """Test character accuracy with text normalization"""
        predictions = ["កម្ពុជា  "]  # Extra spaces
        references = ["កម្ពុជា"]

        result = metrics.calculate_character_accuracy(
            predictions, references, normalize=True
        )

        # Should handle normalization properly
        assert result.score > 0.9  # High accuracy after normalization

    def test_syllable_f1(self, metrics, sample_texts):
        """Test syllable F1 score"""
        result = metrics.calculate_syllable_f1(
            sample_texts['predictions'],
            sample_texts['references']
        )

        assert isinstance(result, KhmerMetricResult)
        assert 0 <= result.score <= 1
        assert 'precision' in result.details
        assert 'recall' in result.details

    def test_word_segmentation_accuracy(self, metrics):
        """Test word segmentation accuracy"""
        predictions = ["ធនាគារ​ជាតិ​កម្ពុជា"]
        references = ["ធនាគារ​ជាតិ​នៃ​កម្ពុជា"]

        result = metrics.calculate_word_segmentation_accuracy(
            predictions, references, use_zwsp=True
        )

        assert isinstance(result, KhmerMetricResult)
        assert 0 <= result.score <= 1
        assert 'boundary_precision' in result.details
        assert 'boundary_recall' in result.details

    def test_edit_distance(self, metrics, sample_texts):
        """Test edit distance calculation"""
        levels = ['char', 'syllable', 'word']

        for level in levels:
            result = metrics.calculate_edit_distance(
                sample_texts['predictions'],
                sample_texts['references'],
                level=level
            )

            assert isinstance(result, KhmerMetricResult)
            assert 0 <= result.score <= 1  # Normalized as similarity
            assert result.details['level'] == level

    def test_code_switching_accuracy(self, metrics):
        """Test code-switching detection accuracy"""
        predictions = ["GDP កម្ពុជា is growing"]
        references = ["GDP កម្ពុជា is developing"]

        result = metrics.calculate_code_switching_accuracy(
            predictions, references
        )

        assert isinstance(result, KhmerMetricResult)
        assert 0 <= result.score <= 1
        assert 'precision' in result.details
        assert 'recall' in result.details

    def test_khmer_specific_errors(self, metrics):
        """Test Khmer-specific error analysis"""
        predictions = ["កមពុជា"]  # Missing subscript
        references = ["កម្ពុជា"]  # With subscript

        result = metrics.calculate_khmer_specific_errors(
            predictions, references
        )

        assert isinstance(result, KhmerMetricResult)
        assert 0 <= result.score <= 1
        assert 'avg_subscript_errors' in result.details
        assert result.details['avg_subscript_errors'] > 0  # Should detect missing subscript

    def test_calculate_all_metrics(self, metrics, sample_texts):
        """Test calculating all metrics at once"""
        results = metrics.calculate_all_metrics(
            sample_texts['predictions'],
            sample_texts['references']
        )

        assert isinstance(results, dict)
        assert len(results) > 5  # Should have multiple metrics

        # Check key metrics are present
        assert 'bleu_syllable' in results
        assert 'character_accuracy' in results
        assert 'syllable_f1' in results

        # All should be KhmerMetricResult instances
        for result in results.values():
            assert isinstance(result, KhmerMetricResult)

    def test_format_results(self, metrics, sample_texts):
        """Test results formatting"""
        results = metrics.calculate_all_metrics(
            sample_texts['predictions'],
            sample_texts['references']
        )

        # Test text format
        text_output = metrics.format_results(results, 'text')
        assert isinstance(text_output, str)
        assert "Khmer Evaluation Metrics" in text_output

        # Test JSON format
        json_output = metrics.format_results(results, 'json')
        assert isinstance(json_output, str)
        import json
        parsed = json.loads(json_output)
        assert isinstance(parsed, dict)

        # Test markdown format
        md_output = metrics.format_results(results, 'markdown')
        assert isinstance(md_output, str)
        assert "# Khmer Evaluation Metrics" in md_output

    def test_empty_inputs(self, metrics):
        """Test handling of empty inputs"""
        with pytest.raises(ValueError):
            metrics.calculate_khmer_bleu([], [])

    def test_mismatched_lengths(self, metrics):
        """Test handling of mismatched input lengths"""
        predictions = ["text1", "text2"]
        references = ["ref1"]

        with pytest.raises(ValueError):
            metrics.calculate_khmer_bleu(predictions, references)

    def test_perfect_match(self, metrics):
        """Test metrics with perfect matches"""
        texts = ["ធនាគារជាតិ", "កម្ពុជា"]

        result = metrics.calculate_character_accuracy(texts, texts)
        assert result.score == 1.0

        result = metrics.calculate_syllable_f1(texts, texts)
        assert result.score == 1.0

    def test_completely_different(self, metrics):
        """Test metrics with completely different texts"""
        predictions = ["ABC"]
        references = ["កម្ពុជា"]

        result = metrics.calculate_character_accuracy(predictions, references)
        assert result.score < 0.1  # Very low accuracy

    def test_metric_consistency(self, metrics, sample_texts):
        """Test that metrics are consistent across runs"""
        result1 = metrics.calculate_khmer_bleu(
            sample_texts['predictions'],
            sample_texts['references']
        )

        result2 = metrics.calculate_khmer_bleu(
            sample_texts['predictions'],
            sample_texts['references']
        )

        assert result1.score == result2.score