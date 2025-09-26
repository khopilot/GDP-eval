"""
Unit tests for Khmer tokenizer
"""

import pytest
import json
from pathlib import Path

from src.metrics.khmer_tokenizer import KhmerTokenizer


class TestKhmerTokenizer:
    """Test Khmer tokenizer functionality"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer instance"""
        return KhmerTokenizer()

    @pytest.fixture
    def test_data(self):
        """Load test data"""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "khmer_test_data.json"
        with open(fixture_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_tokenize_characters(self, tokenizer):
        """Test character tokenization"""
        text = "កម្ពុជា"
        result = tokenizer.tokenize_characters(text)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should handle subscript as cluster
        assert any('្' in token for token in result if len(token) > 1)

    def test_tokenize_syllables(self, tokenizer):
        """Test syllable tokenization"""
        text = "ធនាគារជាតិ"
        result = tokenizer.tokenize_syllables(text)

        assert isinstance(result, list)
        assert len(result) == 5  # Expected syllables
        assert "ធ" in result
        assert "នា" in result

    def test_tokenize_words_with_zwsp(self, tokenizer):
        """Test word tokenization with ZWSP"""
        text = "ធនាគារ​ជាតិ​នៃ​កម្ពុជា"
        result = tokenizer.tokenize_words(text, use_zwsp=True)

        assert isinstance(result, list)
        assert len(result) == 4
        assert "ធនាគារ" in result
        assert "កម្ពុជា" in result

    def test_tokenize_words_without_zwsp(self, tokenizer):
        """Test word tokenization without ZWSP"""
        text = "ធនាគារជាតិ"
        result = tokenizer.tokenize_words(text, use_zwsp=False)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_tokenize_mixed_text(self, tokenizer):
        """Test mixed Khmer-English tokenization"""
        text = "GDP របស់កម្ពុជាគឺ $32 billion"
        result = tokenizer.tokenize_mixed(text)

        assert 'khmer' in result
        assert 'english' in result
        assert 'all' in result
        assert len(result['english']) > 0
        assert 'gdp' in [t.lower() for t in result['english']]

    def test_count_khmer_characters(self, tokenizer):
        """Test Khmer character counting"""
        text = "កម្ពុជា"
        result = tokenizer.count_khmer_characters(text)

        assert 'consonants' in result
        assert 'subscripts' in result
        assert 'total_khmer' in result
        assert result['consonants'] > 0
        assert result['subscripts'] == 1  # One subscript ្

    def test_normalize_khmer(self, tokenizer):
        """Test Khmer text normalization"""
        text = "កម្ពុជា​  ​ជាតិ"  # Extra spaces and ZWSP
        result = tokenizer.normalize_khmer(text)

        assert "​" not in result  # ZWSP removed
        assert "  " not in result  # Multiple spaces normalized

    def test_detect_code_switching(self, tokenizer):
        """Test code-switching detection"""
        text = "GDP កម្ពុជា is growing"
        result = tokenizer.detect_code_switching_points(text)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should detect switches between English and Khmer

    def test_extract_numbers(self, tokenizer):
        """Test number extraction"""
        text = "ឆ្នាំ ២០២៤ and year 2024"
        result = tokenizer.extract_numbers(text)

        assert isinstance(result, list)
        assert len(result) == 2
        # Should find both Khmer and Arabic numbers
        khmer_nums = [n for n in result if n['type'] == 'khmer']
        arabic_nums = [n for n in result if n['type'] == 'arabic']
        assert len(khmer_nums) > 0
        assert len(arabic_nums) > 0

    def test_segment_sentences(self, tokenizer):
        """Test sentence segmentation"""
        text = "ធនាគារជាតិ។ សេដ្ឋកិច្ចកម្ពុជា។"
        result = tokenizer.segment_sentences(text)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all('។' in s for s in result)

    def test_vocabulary_stats(self, tokenizer):
        """Test vocabulary statistics"""
        texts = [
            "ធនាគារជាតិ",
            "ធនាគារពាណិជ្ជ",
            "សេដ្ឋកិច្ច"
        ]
        result = tokenizer.get_vocabulary_stats(texts)

        assert 'unique_words' in result
        assert 'total_words' in result
        assert 'unique_syllables' in result
        assert 'avg_word_length' in result
        assert result['unique_words'] > 0
        assert result['total_words'] >= result['unique_words']

    def test_special_characters(self, tokenizer, test_data):
        """Test handling of special Khmer characters"""
        special = test_data['special_cases']

        # Test subscripts
        subscript_text = special['subscripts']
        chars = tokenizer.tokenize_characters(subscript_text)
        assert any('្' in c for c in chars)

        # Test numerals
        numeral_text = special['numerals']
        numbers = tokenizer.extract_numbers(numeral_text)
        assert len(numbers) > 0

    def test_performance_with_long_text(self, tokenizer):
        """Test performance with longer text"""
        long_text = "កម្ពុជា " * 100  # Repeat 100 times

        # Should handle long text efficiently
        result = tokenizer.tokenize_syllables(long_text)
        assert isinstance(result, list)
        assert len(result) > 0

        # Check caching works
        result2 = tokenizer.tokenize_syllables(long_text[:50])
        assert isinstance(result2, list)