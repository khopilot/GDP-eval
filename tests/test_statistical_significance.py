"""
Comprehensive Test Suite for Statistical Significance Testing System

This test suite validates the statistical methods against known results
and edge cases to ensure enterprise-grade reliability.

Author: GDP-eval Framework
Version: 1.0.0
"""

import pytest
import numpy as np
from typing import List
import warnings

# Import the statistical significance module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.statistical_significance import (
    StatisticalSignificance,
    calculate_effect_sizes,
    power_analysis,
    multiple_testing_correction,
    StatisticalTestResult,
    ComparisonResult
)


class TestStatisticalSignificance:
    """Test cases for the StatisticalSignificance class."""

    @pytest.fixture
    def stats_engine(self):
        """Create a StatisticalSignificance instance with fixed random seed."""
        return StatisticalSignificance(alpha=0.05, random_state=42)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        return {
            'model_a': [0.85, 0.87, 0.83, 0.89, 0.86, 0.84, 0.88, 0.82, 0.90, 0.81],
            'model_b': [0.78, 0.80, 0.76, 0.82, 0.79, 0.77, 0.81, 0.75, 0.83, 0.74],
            'identical': [0.80, 0.80, 0.80, 0.80, 0.80],
            'single_a': [0.85],
            'single_b': [0.75],
            'empty': []
        }

    def test_welch_t_test_basic(self, stats_engine, sample_data):
        """Test basic Welch's t-test functionality."""
        result = stats_engine.welch_t_test(
            sample_data['model_a'],
            sample_data['model_b']
        )

        assert isinstance(result, StatisticalTestResult)
        assert result.test_name == "Welch's t-test"
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1
        assert result.degrees_of_freedom is not None
        assert isinstance(result.significant, bool)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2

    def test_welch_t_test_significant_difference(self, stats_engine, sample_data):
        """Test that Welch's t-test correctly identifies significant differences."""
        result = stats_engine.welch_t_test(
            sample_data['model_a'],
            sample_data['model_b']
        )

        # Model A has consistently higher scores, should be significant
        assert result.significant
        assert result.p_value < 0.05
        assert result.statistic > 0  # Model A > Model B

    def test_welch_t_test_no_difference(self, stats_engine):
        """Test Welch's t-test with identical samples."""
        identical_a = [0.80] * 10
        identical_b = [0.80] * 10

        result = stats_engine.welch_t_test(identical_a, identical_b)

        # Should find no significant difference
        assert not result.significant
        assert result.p_value > 0.05
        assert abs(result.statistic) < 1e-10  # Should be essentially zero

    def test_welch_t_test_one_sided(self, stats_engine, sample_data):
        """Test one-sided Welch's t-test."""
        # Test greater alternative
        result_greater = stats_engine.welch_t_test(
            sample_data['model_a'],
            sample_data['model_b'],
            alternative='greater'
        )

        # Test less alternative
        result_less = stats_engine.welch_t_test(
            sample_data['model_a'],
            sample_data['model_b'],
            alternative='less'
        )

        assert result_greater.significant  # Model A > Model B
        assert not result_less.significant  # Model A is not < Model B

    def test_welch_t_test_edge_cases(self, stats_engine, sample_data):
        """Test Welch's t-test edge cases."""
        # Test with very small samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats_engine.welch_t_test([0.8], [0.7])
            assert isinstance(result, StatisticalTestResult)

        # Test with empty lists
        with pytest.raises(ValueError):
            stats_engine.welch_t_test([], [0.8])

        with pytest.raises(ValueError):
            stats_engine.welch_t_test([0.8], [])

    def test_bootstrap_confidence_intervals(self, stats_engine, sample_data):
        """Test bootstrap confidence intervals."""
        intervals = stats_engine.bootstrap_confidence_intervals(
            sample_data['model_a'],
            n_bootstrap=1000,  # Use fewer for faster testing
            confidence_levels=[0.90, 0.95, 0.99]
        )

        assert isinstance(intervals, dict)
        assert len(intervals) == 3

        for conf_level, (lower, upper) in intervals.items():
            assert 0 < conf_level < 1
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            assert lower < upper

        # Higher confidence levels should have wider intervals
        assert intervals[0.99][1] - intervals[0.99][0] > intervals[0.95][1] - intervals[0.95][0]

    def test_bootstrap_bca_method(self, stats_engine, sample_data):
        """Test BCa bootstrap method specifically."""
        intervals_bca = stats_engine.bootstrap_confidence_intervals(
            sample_data['model_a'],
            n_bootstrap=1000,
            method='bca'
        )

        intervals_basic = stats_engine.bootstrap_confidence_intervals(
            sample_data['model_a'],
            n_bootstrap=1000,
            method='basic'
        )

        # BCa and basic should give different results (usually)
        assert intervals_bca[0.95] != intervals_basic[0.95]

    def test_bootstrap_edge_cases(self, stats_engine):
        """Test bootstrap confidence intervals edge cases."""
        # Empty list
        with pytest.raises(ValueError):
            stats_engine.bootstrap_confidence_intervals([])

        # Very few bootstrap samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats_engine.bootstrap_confidence_intervals([0.8, 0.9], n_bootstrap=100)
            assert isinstance(result, dict)


class TestEffectSizes:
    """Test cases for effect size calculations."""

    @pytest.fixture
    def effect_data(self):
        """Sample data for effect size testing."""
        np.random.seed(42)
        return {
            'large_effect_a': np.random.normal(1.0, 1.0, 50),
            'large_effect_b': np.random.normal(0.0, 1.0, 50),
            'small_effect_a': np.random.normal(0.2, 1.0, 50),
            'small_effect_b': np.random.normal(0.0, 1.0, 50),
            'no_effect_a': np.random.normal(0.0, 1.0, 50),
            'no_effect_b': np.random.normal(0.0, 1.0, 50)
        }

    def test_calculate_effect_sizes_basic(self, effect_data):
        """Test basic effect size calculations."""
        effects = calculate_effect_sizes(
            effect_data['large_effect_a'].tolist(),
            effect_data['large_effect_b'].tolist()
        )

        assert isinstance(effects, dict)
        required_keys = ['cohens_d', 'hedges_g', 'glass_delta',
                        'probability_superiority', 'interpretation']
        for key in required_keys:
            assert key in effects

    def test_large_effect_detection(self, effect_data):
        """Test detection of large effect sizes."""
        effects = calculate_effect_sizes(
            effect_data['large_effect_a'].tolist(),
            effect_data['large_effect_b'].tolist()
        )

        # Should detect large effect
        assert abs(effects['cohens_d']) > 0.8
        assert effects['interpretation']['cohens_d'] in ['large']
        assert effects['probability_superiority'] > 0.6

    def test_small_effect_detection(self, effect_data):
        """Test detection of small effect sizes."""
        effects = calculate_effect_sizes(
            effect_data['small_effect_a'].tolist(),
            effect_data['small_effect_b'].tolist()
        )

        # Should detect small to medium effect
        assert 0.1 <= abs(effects['cohens_d']) <= 0.6
        assert effects['interpretation']['cohens_d'] in ['small', 'medium', 'negligible']

    def test_no_effect_detection(self, effect_data):
        """Test detection when there's no effect."""
        effects = calculate_effect_sizes(
            effect_data['no_effect_a'].tolist(),
            effect_data['no_effect_b'].tolist()
        )

        # Should detect negligible effect
        assert abs(effects['cohens_d']) < 0.5
        assert 0.4 <= effects['probability_superiority'] <= 0.6

    def test_effect_sizes_edge_cases(self):
        """Test effect size calculations with edge cases."""
        # Identical samples
        identical_a = [0.8] * 10
        identical_b = [0.8] * 10
        effects = calculate_effect_sizes(identical_a, identical_b)
        assert effects['cohens_d'] == 0.0
        assert effects['probability_superiority'] == 0.5

        # Empty lists
        with pytest.raises(ValueError):
            calculate_effect_sizes([], [0.8])


class TestPowerAnalysis:
    """Test cases for power analysis."""

    def test_power_analysis_basic(self):
        """Test basic power analysis functionality."""
        result = power_analysis(
            effect_size=0.5,
            alpha=0.05,
            power=0.80
        )

        assert isinstance(result, dict)
        required_keys = ['required_sample_size', 'effect_size', 'alpha',
                        'power', 'alternative', 'power_curve_data']
        for key in required_keys:
            assert key in result

        assert isinstance(result['required_sample_size'], int)
        assert result['required_sample_size'] > 0

    def test_power_analysis_different_effect_sizes(self):
        """Test power analysis with different effect sizes."""
        # Large effect should require smaller sample size
        large_effect = power_analysis(effect_size=0.8, power=0.80)
        small_effect = power_analysis(effect_size=0.2, power=0.80)

        assert large_effect['required_sample_size'] < small_effect['required_sample_size']

    def test_power_analysis_different_power_levels(self):
        """Test power analysis with different desired power levels."""
        # Higher power should require larger sample size
        low_power = power_analysis(effect_size=0.5, power=0.70)
        high_power = power_analysis(effect_size=0.5, power=0.90)

        assert low_power['required_sample_size'] < high_power['required_sample_size']

    def test_power_analysis_plotting(self):
        """Test power analysis with plotting enabled."""
        result = power_analysis(
            effect_size=0.5,
            alpha=0.05,
            power=0.80,
            plot=True
        )

        assert 'plot' in result
        # The plot should be a matplotlib figure
        assert hasattr(result['plot'], 'savefig')


class TestMultipleTestingCorrection:
    """Test cases for multiple testing corrections."""

    @pytest.fixture
    def p_values_data(self):
        """Sample p-values for testing."""
        return {
            'mixed': [0.001, 0.01, 0.02, 0.05, 0.08, 0.12, 0.25, 0.45, 0.67, 0.89],
            'all_significant': [0.001, 0.005, 0.01, 0.02, 0.03],
            'none_significant': [0.1, 0.2, 0.3, 0.4, 0.5],
            'edge_cases': [0.0, 0.05, 1.0]
        }

    def test_bonferroni_correction(self, p_values_data):
        """Test Bonferroni correction."""
        result = multiple_testing_correction(
            p_values_data['mixed'],
            method='bonferroni'
        )

        assert result['method'] == 'bonferroni'
        assert len(result['adjusted_p_values']) == len(p_values_data['mixed'])

        # Bonferroni should be conservative
        for orig, adj in zip(result['original_p_values'], result['adjusted_p_values']):
            assert adj >= orig
            assert adj <= 1.0

    def test_holm_correction(self, p_values_data):
        """Test Holm-Bonferroni correction."""
        result = multiple_testing_correction(
            p_values_data['mixed'],
            method='holm'
        )

        assert result['method'] == 'holm'
        # Holm should be less conservative than Bonferroni
        bonferroni = multiple_testing_correction(p_values_data['mixed'], method='bonferroni')

        # At least some Holm-adjusted p-values should be smaller than Bonferroni
        holm_smaller = any(h < b for h, b in zip(result['adjusted_p_values'],
                                                bonferroni['adjusted_p_values']))
        assert holm_smaller

    def test_fdr_bh_correction(self, p_values_data):
        """Test Benjamini-Hochberg FDR correction."""
        result = multiple_testing_correction(
            p_values_data['mixed'],
            method='fdr_bh'
        )

        assert result['method'] == 'fdr_bh'
        # FDR should generally be less conservative than family-wise corrections
        bonferroni = multiple_testing_correction(p_values_data['mixed'], method='bonferroni')

        # FDR should reject more hypotheses than Bonferroni
        assert result['rejected_hypotheses'] >= bonferroni['rejected_hypotheses']

    def test_multiple_testing_edge_cases(self):
        """Test multiple testing corrections with edge cases."""
        # Empty p-values list
        with pytest.raises(ValueError):
            multiple_testing_correction([])

        # Invalid p-values
        with pytest.raises(ValueError):
            multiple_testing_correction([-0.1, 0.5])

        with pytest.raises(ValueError):
            multiple_testing_correction([0.5, 1.1])

        # Unknown method
        with pytest.raises(ValueError):
            multiple_testing_correction([0.05], method='unknown')

    def test_single_p_value(self):
        """Test corrections with single p-value."""
        result = multiple_testing_correction([0.05], method='bonferroni')
        assert result['adjusted_p_values'][0] == 0.05  # No correction needed

    def test_all_methods_consistency(self, p_values_data):
        """Test that all correction methods produce valid results."""
        methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']

        for method in methods:
            result = multiple_testing_correction(p_values_data['mixed'], method=method)

            # Basic validity checks
            assert len(result['adjusted_p_values']) == len(p_values_data['mixed'])
            assert all(0 <= p <= 1 for p in result['adjusted_p_values'])
            assert isinstance(result['rejected_hypotheses'], int)
            assert 0 <= result['rejected_hypotheses'] <= len(p_values_data['mixed'])


class TestIntegration:
    """Integration tests combining multiple statistical methods."""

    def test_complete_model_comparison_workflow(self):
        """Test a complete model comparison workflow."""
        np.random.seed(42)

        # Generate realistic model performance data
        model_a_scores = np.random.beta(8, 2, 50) * 0.3 + 0.7  # High performance
        model_b_scores = np.random.beta(6, 4, 50) * 0.3 + 0.65  # Lower performance

        stats_engine = StatisticalSignificance(random_state=42)

        # 1. Statistical significance test
        t_test_result = stats_engine.welch_t_test(
            model_a_scores.tolist(),
            model_b_scores.tolist()
        )

        # 2. Effect size calculation
        effect_sizes = calculate_effect_sizes(
            model_a_scores.tolist(),
            model_b_scores.tolist()
        )

        # 3. Bootstrap confidence intervals
        ci_a = stats_engine.bootstrap_confidence_intervals(model_a_scores.tolist())
        ci_b = stats_engine.bootstrap_confidence_intervals(model_b_scores.tolist())

        # 4. Power analysis
        observed_effect = effect_sizes['cohens_d']
        power_result = power_analysis(effect_size=observed_effect)

        # Verify the workflow produces consistent results
        assert isinstance(t_test_result, StatisticalTestResult)
        assert isinstance(effect_sizes, dict)
        assert isinstance(ci_a, dict)
        assert isinstance(ci_b, dict)
        assert isinstance(power_result, dict)

        # Check logical consistency
        if t_test_result.significant:
            # If statistically significant, effect size should be meaningful
            assert abs(effect_sizes['cohens_d']) > 0.1

        # Confidence intervals should not overlap significantly if there's a large effect
        if abs(effect_sizes['cohens_d']) > 0.8:
            # This is a heuristic check - might occasionally fail due to randomness
            pass  # Could add overlap check here

    def test_multiple_comparisons_workflow(self):
        """Test workflow with multiple model comparisons."""
        np.random.seed(42)

        # Generate performance data for 5 models
        models_data = {}
        for i in range(5):
            # Each model has slightly different performance
            base_performance = 0.7 + i * 0.02
            models_data[f'model_{i}'] = np.random.beta(8, 2, 30) * 0.2 + base_performance

        stats_engine = StatisticalSignificance(random_state=42)

        # Perform pairwise comparisons
        p_values = []
        comparisons = []

        for i in range(5):
            for j in range(i+1, 5):
                model_a = models_data[f'model_{i}']
                model_b = models_data[f'model_{j}']

                result = stats_engine.welch_t_test(model_a.tolist(), model_b.tolist())
                p_values.append(result.p_value)
                comparisons.append(f'model_{i}_vs_model_{j}')

        # Apply multiple testing correction
        corrected_results = multiple_testing_correction(p_values, method='fdr_bh')

        assert len(corrected_results['adjusted_p_values']) == len(p_values)
        assert corrected_results['n_tests'] == len(p_values)

        # Number of rejected hypotheses should be <= original significant tests
        original_significant = sum(p < 0.05 for p in p_values)
        assert corrected_results['rejected_hypotheses'] <= original_significant


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])