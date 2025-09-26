"""
Statistical Significance Testing System for GDP-eval Framework

This module provides enterprise-grade statistical testing capabilities for comparing
AI model performance with OpenAI/Anthropic-level rigor. Designed specifically for
evaluating AI models' impact on Cambodia's economy.

Features:
- Welch's t-test for unequal variances
- Bootstrap confidence intervals with BCa method
- Multiple effect size calculations
- Power analysis with visualizations
- Multiple testing corrections

Author: GDP-eval Framework
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# Configure logging
logger = logging.getLogger(__name__)


class StatisticalTest(Enum):
    """Enumeration of supported statistical tests."""
    WELCH_T_TEST = "welch_t"
    MANN_WHITNEY_U = "mann_whitney"
    PERMUTATION_TEST = "permutation"


class MultipleTestingMethod(Enum):
    """Enumeration of multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    FDR_BH = "fdr_bh"
    FDR_BY = "fdr_by"


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[float] = None
    confidence_level: float = 0.95
    significant: bool = False
    interpretation: str = ""
    effect_size: Optional[Dict] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class ComparisonResult:
    """Container for comprehensive model comparison results."""
    model_a_name: str
    model_b_name: str
    sample_sizes: Tuple[int, int]
    descriptive_stats: Dict
    statistical_tests: List[StatisticalTestResult]
    effect_sizes: Dict
    confidence_intervals: Dict
    power_analysis: Optional[Dict] = None
    recommendation: str = ""


class StatisticalSignificance:
    """
    Enterprise-grade statistical significance testing system.

    This class provides comprehensive statistical testing capabilities for
    comparing AI model performance with rigorous methodology.
    """

    def __init__(self, alpha: float = 0.05, random_state: Optional[int] = None):
        """
        Initialize the statistical testing system.

        Args:
            alpha: Significance level (default: 0.05)
            random_state: Random seed for reproducible results
        """
        self.alpha = alpha
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def welch_t_test(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        alternative: str = "two-sided",
        confidence_level: float = 0.95
    ) -> StatisticalTestResult:
        """
        Perform Welch's t-test for unequal variances.

        Welch's t-test is more robust than Student's t-test when sample sizes
        or variances are unequal, making it ideal for comparing AI model
        performance where evaluation conditions may vary.

        Args:
            model_a_scores: Performance scores for model A
            model_b_scores: Performance scores for model B
            alternative: Type of alternative hypothesis
                - 'two-sided': means are unequal
                - 'less': mean of A is less than B
                - 'greater': mean of A is greater than B
            confidence_level: Confidence level for interpretation

        Returns:
            StatisticalTestResult with detailed test outcomes

        Raises:
            ValueError: If input data is invalid

        References:
            Welch, B.L. (1947). "The generalization of 'Student's' problem when several
            different population variances are involved." Biometrika, 34(1-2), 28-35.
        """
        # Input validation
        if not model_a_scores or not model_b_scores:
            raise ValueError("Both score lists must contain at least one value")

        if len(model_a_scores) < 2 or len(model_b_scores) < 2:
            warnings.warn("Sample sizes are very small (n<2). Results may be unreliable.")

        # Convert to numpy arrays
        a_scores = np.array(model_a_scores, dtype=float)
        b_scores = np.array(model_b_scores, dtype=float)

        # Check for constant values
        if np.var(a_scores) == 0 and np.var(b_scores) == 0:
            warnings.warn("Both samples have zero variance. Test may be inappropriate.")

        try:
            # Perform Welch's t-test
            statistic, p_value = stats.ttest_ind(
                a_scores, b_scores,
                equal_var=False,  # This makes it Welch's t-test
                alternative=alternative
            )

            # Calculate degrees of freedom using Welch-Satterthwaite equation
            n_a, n_b = len(a_scores), len(b_scores)
            var_a, var_b = np.var(a_scores, ddof=1), np.var(b_scores, ddof=1)

            # Welch-Satterthwaite degrees of freedom
            if var_a == 0 and var_b == 0:
                df = n_a + n_b - 2  # Fallback to pooled df
            else:
                numerator = (var_a/n_a + var_b/n_b) ** 2
                denominator = (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
                df = numerator / denominator if denominator > 0 else n_a + n_b - 2

            # Determine significance
            significant = p_value < self.alpha

            # Create interpretation
            mean_a, mean_b = np.mean(a_scores), np.mean(b_scores)
            interpretation = self._create_welch_interpretation(
                statistic, p_value, mean_a, mean_b, significant, alternative
            )

            # Calculate confidence interval for difference in means
            ci_lower, ci_upper = self._welch_confidence_interval(
                a_scores, b_scores, confidence_level
            )

            return StatisticalTestResult(
                test_name="Welch's t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                degrees_of_freedom=float(df),
                confidence_level=confidence_level,
                significant=significant,
                interpretation=interpretation,
                confidence_interval=(ci_lower, ci_upper)
            )

        except Exception as e:
            logger.error(f"Error in Welch's t-test: {e}")
            raise ValueError(f"Failed to perform Welch's t-test: {e}")

    def _welch_confidence_interval(
        self,
        a_scores: np.ndarray,
        b_scores: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means using Welch's method."""
        n_a, n_b = len(a_scores), len(b_scores)
        mean_a, mean_b = np.mean(a_scores), np.mean(b_scores)
        var_a, var_b = np.var(a_scores, ddof=1), np.var(b_scores, ddof=1)

        # Standard error of difference
        se_diff = np.sqrt(var_a/n_a + var_b/n_b)

        # Degrees of freedom (Welch-Satterthwaite)
        if var_a == 0 and var_b == 0:
            df = n_a + n_b - 2
        else:
            numerator = (var_a/n_a + var_b/n_b) ** 2
            denominator = (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
            df = numerator / denominator if denominator > 0 else n_a + n_b - 2

        # Critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)

        # Confidence interval for difference (A - B)
        diff = mean_a - mean_b
        margin_error = t_critical * se_diff

        return (diff - margin_error, diff + margin_error)

    def _create_welch_interpretation(
        self,
        statistic: float,
        p_value: float,
        mean_a: float,
        mean_b: float,
        significant: bool,
        alternative: str
    ) -> str:
        """Create human-readable interpretation of Welch's t-test results."""
        if significant:
            if alternative == "two-sided":
                direction = "significantly higher" if mean_a > mean_b else "significantly lower"
                return (f"Model A performs {direction} than Model B "
                       f"(t={statistic:.3f}, p={p_value:.4f}). "
                       f"The difference is statistically significant at α={self.alpha}.")
            elif alternative == "greater":
                if statistic > 0:
                    return (f"Model A performs significantly better than Model B "
                           f"(t={statistic:.3f}, p={p_value:.4f}).")
                else:
                    return (f"No evidence that Model A performs better than Model B "
                           f"(t={statistic:.3f}, p={p_value:.4f}).")
            elif alternative == "less":
                if statistic < 0:
                    return (f"Model A performs significantly worse than Model B "
                           f"(t={statistic:.3f}, p={p_value:.4f}).")
                else:
                    return (f"No evidence that Model A performs worse than Model B "
                           f"(t={statistic:.3f}, p={p_value:.4f}).")
        else:
            return (f"No significant difference between Model A and Model B "
                   f"(t={statistic:.3f}, p={p_value:.4f}). "
                   f"The performance difference could be due to random variation.")

    def bootstrap_confidence_intervals(
        self,
        scores: List[float],
        n_bootstrap: int = 10000,
        confidence_levels: List[float] = [0.95, 0.99],
        statistic_func: callable = np.mean,
        method: str = "bca"
    ) -> Dict[float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals using BCa method.

        The bias-corrected and accelerated (BCa) bootstrap method provides
        more accurate confidence intervals than basic bootstrap, especially
        for skewed distributions common in AI model evaluation.

        Args:
            scores: Sample data
            n_bootstrap: Number of bootstrap samples
            confidence_levels: List of confidence levels to compute
            statistic_func: Function to compute statistic (default: mean)
            method: Bootstrap method ('basic', 'percentile', 'bca')

        Returns:
            Dictionary mapping confidence levels to (lower, upper) bounds

        References:
            Efron, B., & Tibshirani, R. J. (1993). "An Introduction to the Bootstrap."
            Chapman & Hall/CRC.
        """
        if not scores:
            raise ValueError("Scores list cannot be empty")

        if n_bootstrap < 1000:
            warnings.warn("n_bootstrap < 1000 may yield unreliable confidence intervals")

        scores_array = np.array(scores, dtype=float)
        n = len(scores_array)

        # Original statistic
        original_stat = statistic_func(scores_array)

        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = resample(scores_array, n_samples=n, random_state=None)
            bootstrap_stats.append(statistic_func(bootstrap_sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence intervals for each level
        intervals = {}

        for conf_level in confidence_levels:
            alpha = 1 - conf_level

            if method == "basic":
                # Basic bootstrap method
                lower = np.percentile(bootstrap_stats, 100 * alpha/2)
                upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))

            elif method == "percentile":
                # Percentile method
                lower = np.percentile(bootstrap_stats, 100 * alpha/2)
                upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))

            elif method == "bca":
                # BCa method (more complex but more accurate)
                lower, upper = self._bca_confidence_interval(
                    scores_array, bootstrap_stats, original_stat,
                    conf_level, statistic_func
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            intervals[conf_level] = (float(lower), float(upper))

        return intervals

    def _bca_confidence_interval(
        self,
        original_sample: np.ndarray,
        bootstrap_stats: np.ndarray,
        original_stat: float,
        conf_level: float,
        statistic_func: callable
    ) -> Tuple[float, float]:
        """Calculate BCa confidence interval."""
        n = len(original_sample)
        alpha = 1 - conf_level

        # Bias correction
        n_less = np.sum(bootstrap_stats < original_stat)
        z0 = stats.norm.ppf(n_less / len(bootstrap_stats)) if n_less > 0 else 0

        # Acceleration correction using jackknife
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([original_sample[:i], original_sample[i+1:]])
            jackknife_stats.append(statistic_func(jackknife_sample))

        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)

        # Acceleration parameter
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** 1.5
        a = numerator / denominator if denominator != 0 else 0

        # Adjusted percentiles
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2)))

        # Ensure percentiles are within valid range
        alpha1 = max(0, min(1, alpha1))
        alpha2 = max(0, min(1, alpha2))

        lower = np.percentile(bootstrap_stats, 100 * alpha1)
        upper = np.percentile(bootstrap_stats, 100 * alpha2)

        return lower, upper


def calculate_effect_sizes(model_a: List[float], model_b: List[float]) -> Dict:
    """
    Calculate multiple effect size measures.

    Effect sizes quantify the magnitude of difference between groups,
    providing practical significance beyond statistical significance.

    Args:
        model_a: Performance scores for model A
        model_b: Performance scores for model B

    Returns:
        Dictionary containing various effect size measures

    References:
        Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences."
        Lawrence Erlbaum Associates.
    """
    if not model_a or not model_b:
        raise ValueError("Both score lists must contain at least one value")

    a_scores = np.array(model_a, dtype=float)
    b_scores = np.array(model_b, dtype=float)

    n_a, n_b = len(a_scores), len(b_scores)
    mean_a, mean_b = np.mean(a_scores), np.mean(b_scores)
    var_a, var_b = np.var(a_scores, ddof=1), np.var(b_scores, ddof=1)

    effect_sizes = {}

    # Cohen's d (standardized mean difference)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
    effect_sizes['cohens_d'] = float(cohens_d)

    # Hedge's g (corrected for small samples)
    correction_factor = 1 - (3 / (4 * (n_a + n_b) - 9))
    hedges_g = cohens_d * correction_factor
    effect_sizes['hedges_g'] = float(hedges_g)

    # Glass's delta (using control group standard deviation)
    glass_delta = (mean_a - mean_b) / np.sqrt(var_b) if var_b > 0 else 0
    effect_sizes['glass_delta'] = float(glass_delta)

    # Probability of superiority (A12 statistic)
    prob_superiority = np.mean([a > b for a in a_scores for b in b_scores])
    effect_sizes['probability_superiority'] = float(prob_superiority)

    # Effect size interpretations
    effect_sizes['interpretation'] = {
        'cohens_d': _interpret_cohens_d(cohens_d),
        'hedges_g': _interpret_cohens_d(hedges_g),  # Same interpretation as Cohen's d
        'glass_delta': _interpret_cohens_d(glass_delta),
        'probability_superiority': _interpret_probability_superiority(prob_superiority)
    }

    return effect_sizes


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def _interpret_probability_superiority(prob: float) -> str:
    """Interpret probability of superiority."""
    if prob < 0.44:
        return "Model B clearly superior"
    elif prob < 0.47:
        return "Model B probably superior"
    elif prob < 0.53:
        return "Models roughly equivalent"
    elif prob < 0.56:
        return "Model A probably superior"
    else:
        return "Model A clearly superior"


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    plot: bool = False,
    save_plot: Optional[str] = None
) -> Dict:
    """
    Perform power analysis to determine required sample size.

    Power analysis helps determine the minimum sample size needed to detect
    an effect of a given size with specified probability (power).

    Args:
        effect_size: Expected effect size (Cohen's d)
        alpha: Type I error rate (significance level)
        power: Desired statistical power (1 - Type II error rate)
        alternative: Type of test ('two-sided', 'larger', 'smaller')
        plot: Whether to generate power curve visualization
        save_plot: File path to save the plot

    Returns:
        Dictionary containing power analysis results

    References:
        Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences."
    """
    try:
        from statsmodels.stats.power import ttest_power

        # Calculate required sample size
        required_n = ttest_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            alternative=alternative
        )

        # Calculate power for different sample sizes
        sample_sizes = np.arange(5, 200, 5)
        powers = [
            ttest_power(
                effect_size=effect_size,
                nobs=n,
                alpha=alpha,
                alternative=alternative
            ) for n in sample_sizes
        ]

        results = {
            'required_sample_size': int(np.ceil(required_n)),
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power,
            'alternative': alternative,
            'power_curve_data': {
                'sample_sizes': sample_sizes.tolist(),
                'powers': powers
            }
        }

        # Generate power curve plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot power curve
            ax.plot(sample_sizes, powers, 'b-', linewidth=2, label='Power curve')

            # Add reference lines
            ax.axhline(y=power, color='r', linestyle='--', alpha=0.7,
                      label=f'Desired power = {power}')
            ax.axvline(x=required_n, color='g', linestyle='--', alpha=0.7,
                      label=f'Required n = {int(np.ceil(required_n))}')

            # Formatting
            ax.set_xlabel('Sample Size (n per group)')
            ax.set_ylabel('Statistical Power')
            ax.set_title(f'Power Analysis (Effect size = {effect_size}, α = {alpha})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1)

            # Add interpretation text
            interpretation = (
                f"To detect an effect size of {effect_size} with {power*100}% power "
                f"at α = {alpha}, you need at least {int(np.ceil(required_n))} "
                f"observations per group."
            )
            ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

            plt.tight_layout()

            if save_plot:
                plt.savefig(save_plot, dpi=300, bbox_inches='tight')
                logger.info(f"Power analysis plot saved to {save_plot}")

            results['plot'] = fig

        return results

    except Exception as e:
        logger.error(f"Error in power analysis: {e}")
        raise ValueError(f"Failed to perform power analysis: {e}")


def multiple_testing_correction(
    p_values: List[float],
    method: str = "bonferroni",
    alpha: float = 0.05
) -> Dict:
    """
    Correct p-values for multiple hypothesis testing.

    When conducting multiple statistical tests, the probability of making
    at least one Type I error (false positive) increases. Multiple testing
    corrections control the family-wise error rate or false discovery rate.

    Args:
        p_values: List of p-values from multiple tests
        method: Correction method
            - 'bonferroni': Bonferroni correction (most conservative)
            - 'holm': Holm-Bonferroni method (less conservative)
            - 'fdr_bh': Benjamini-Hochberg FDR (controls false discovery rate)
            - 'fdr_by': Benjamini-Yekutieli FDR (for dependent tests)
        alpha: Family-wise error rate or false discovery rate

    Returns:
        Dictionary with corrected p-values and significance decisions

    References:
        Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate."
        Journal of the Royal Statistical Society, 57(1), 289-300.
    """
    if not p_values:
        raise ValueError("p_values list cannot be empty")

    p_array = np.array(p_values, dtype=float)
    n_tests = len(p_array)

    # Validate p-values
    if np.any(p_array < 0) or np.any(p_array > 1):
        raise ValueError("All p-values must be between 0 and 1")

    results = {
        'method': method,
        'n_tests': n_tests,
        'alpha': alpha,
        'original_p_values': p_values.copy(),
        'adjusted_p_values': [],
        'significant': [],
        'rejected_hypotheses': 0
    }

    if method == "bonferroni":
        # Bonferroni correction: multiply each p-value by number of tests
        adjusted_p = np.minimum(p_array * n_tests, 1.0)

    elif method == "holm":
        # Holm-Bonferroni method (step-down procedure)
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        adjusted_p = np.zeros_like(p_array)
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            # Holm adjustment: p * (n - i)
            adjusted_p[idx] = min(p * (n_tests - i), 1.0)

        # Ensure monotonicity (each adjusted p >= previous)
        sorted_adj_p = adjusted_p[sorted_indices]
        for i in range(1, len(sorted_adj_p)):
            if sorted_adj_p[i] < sorted_adj_p[i-1]:
                sorted_adj_p[i] = sorted_adj_p[i-1]

        # Map back to original order
        for i, idx in enumerate(sorted_indices):
            adjusted_p[idx] = sorted_adj_p[i]

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR control
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        adjusted_p = np.zeros_like(p_array)
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            # BH adjustment: p * n / (i + 1)
            adjusted_p[idx] = min(p * n_tests / (i + 1), 1.0)

        # Ensure monotonicity (reverse)
        sorted_adj_p = adjusted_p[sorted_indices]
        for i in range(len(sorted_adj_p) - 2, -1, -1):
            if sorted_adj_p[i] > sorted_adj_p[i+1]:
                sorted_adj_p[i] = sorted_adj_p[i+1]

        # Map back to original order
        for i, idx in enumerate(sorted_indices):
            adjusted_p[idx] = sorted_adj_p[i]

    elif method == "fdr_by":
        # Benjamini-Yekutieli FDR (for dependent tests)
        harmonic_sum = np.sum(1.0 / np.arange(1, n_tests + 1))
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        adjusted_p = np.zeros_like(p_array)
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            # BY adjustment: p * n * harmonic_sum / (i + 1)
            adjusted_p[idx] = min(p * n_tests * harmonic_sum / (i + 1), 1.0)

        # Ensure monotonicity (reverse)
        sorted_adj_p = adjusted_p[sorted_indices]
        for i in range(len(sorted_adj_p) - 2, -1, -1):
            if sorted_adj_p[i] > sorted_adj_p[i+1]:
                sorted_adj_p[i] = sorted_adj_p[i+1]

        # Map back to original order
        for i, idx in enumerate(sorted_indices):
            adjusted_p[idx] = sorted_adj_p[i]

    else:
        raise ValueError(f"Unknown correction method: {method}")

    # Determine significance
    significant = adjusted_p <= alpha
    rejected_hypotheses = np.sum(significant)

    results.update({
        'adjusted_p_values': adjusted_p.tolist(),
        'significant': significant.tolist(),
        'rejected_hypotheses': int(rejected_hypotheses),
        'power_estimate': rejected_hypotheses / n_tests if n_tests > 0 else 0.0
    })

    # Add interpretation
    if method in ['bonferroni', 'holm']:
        error_type = "family-wise error rate"
    else:
        error_type = "false discovery rate"

    results['interpretation'] = (
        f"Using {method} correction with α={alpha}, {rejected_hypotheses} out of "
        f"{n_tests} hypotheses were rejected, controlling the {error_type}."
    )

    return results


if __name__ == "__main__":
    # Example usage for testing
    import doctest
    doctest.testmod()

    # Simple demonstration
    np.random.seed(42)
    model_a_scores = [0.85, 0.87, 0.83, 0.89, 0.86, 0.84, 0.88, 0.82]
    model_b_scores = [0.78, 0.80, 0.76, 0.82, 0.79, 0.77, 0.81, 0.75]

    stats_engine = StatisticalSignificance(random_state=42)

    # Welch's t-test
    result = stats_engine.welch_t_test(model_a_scores, model_b_scores)
    print(f"Welch's t-test: {result.interpretation}")

    # Bootstrap confidence intervals
    intervals = stats_engine.bootstrap_confidence_intervals(model_a_scores)
    print(f"95% CI for Model A: {intervals[0.95]}")

    # Effect sizes
    effects = calculate_effect_sizes(model_a_scores, model_b_scores)
    print(f"Cohen's d: {effects['cohens_d']:.3f} ({effects['interpretation']['cohens_d']})")

    # Power analysis
    power_result = power_analysis(effect_size=0.8, alpha=0.05, power=0.80)
    print(f"Required sample size: {power_result['required_sample_size']} per group")

    # Multiple testing correction
    p_vals = [0.01, 0.05, 0.02, 0.10, 0.001]
    correction_result = multiple_testing_correction(p_vals, method="bonferroni")
    print(f"Bonferroni correction: {correction_result['interpretation']}")