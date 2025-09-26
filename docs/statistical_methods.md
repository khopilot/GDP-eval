# Statistical Methods Documentation

## Overview

The GDP-eval Statistical Significance Testing System provides enterprise-grade statistical analysis for comparing AI model performance with OpenAI/Anthropic-level rigor. This documentation covers the theoretical foundations, practical applications, and interpretation guidelines for each statistical method.

## Table of Contents

1. [Welch's t-test](#welchs-t-test)
2. [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
3. [Effect Size Calculations](#effect-size-calculations)
4. [Power Analysis](#power-analysis)
5. [Multiple Testing Corrections](#multiple-testing-corrections)
6. [Integration Examples](#integration-examples)
7. [References](#references)

## Welch's t-test

### Purpose
Welch's t-test compares the means of two independent samples without assuming equal variances or sample sizes. This makes it more robust than Student's t-test for real-world AI model evaluations where conditions may vary.

### When to Use
- Comparing performance scores between two AI models
- Sample sizes are different between models
- Variances might be unequal (common in AI evaluation)
- Data is approximately normally distributed

### Mathematical Foundation

The Welch's t-test statistic is:

```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

Where:
- x̄₁, x̄₂ = sample means
- s₁², s₂² = sample variances
- n₁, n₂ = sample sizes

The degrees of freedom use the Welch-Satterthwaite equation:

```
df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

### Interpretation Guidelines

| p-value Range | Interpretation | Action |
|---------------|----------------|---------|
| p < 0.001 | Very strong evidence of difference | High confidence in model superiority |
| 0.001 ≤ p < 0.01 | Strong evidence | Good confidence in difference |
| 0.01 ≤ p < 0.05 | Moderate evidence | Significant at α = 0.05 |
| 0.05 ≤ p < 0.1 | Weak evidence | Consider practical significance |
| p ≥ 0.1 | Little/no evidence | Models likely equivalent |

### Example Usage

```python
from src.evaluation.statistical_significance import StatisticalSignificance

stats = StatisticalSignificance(alpha=0.05)

# Model performance scores
model_a_scores = [0.85, 0.87, 0.83, 0.89, 0.86]
model_b_scores = [0.78, 0.80, 0.76, 0.82, 0.79]

result = stats.welch_t_test(model_a_scores, model_b_scores)
print(f"Result: {result.interpretation}")
```

### Assumptions and Limitations

**Assumptions:**
- Independence of observations
- Approximately normal distribution (robust to violations with n > 30)
- Continuous data

**Limitations:**
- Sensitive to extreme outliers
- May have low power with very small samples (n < 5)
- Assumes observations are representative of populations

## Bootstrap Confidence Intervals

### Purpose
Bootstrap confidence intervals provide robust estimation of uncertainty around statistics without requiring distributional assumptions. The BCa (bias-corrected and accelerated) method provides superior accuracy for skewed distributions.

### When to Use
- Non-normal or unknown distributions
- Small sample sizes
- Robust uncertainty quantification
- When parametric methods are questionable

### Mathematical Foundation

**Basic Bootstrap Process:**
1. Resample with replacement from original data n times
2. Calculate statistic for each bootstrap sample
3. Use distribution of bootstrap statistics for inference

**BCa Correction:**
- **Bias Correction (z₀):** Adjusts for bias in bootstrap distribution
- **Acceleration (a):** Corrects for skewness using jackknife

```
α₁ = Φ(z₀ + (z₀ + z_{α/2})/(1 - a(z₀ + z_{α/2})))
α₂ = Φ(z₀ + (z₀ + z_{1-α/2})/(1 - a(z₀ + z_{1-α/2})))
```

### Confidence Level Interpretation

| Confidence Level | Coverage | Use Case |
|------------------|----------|----------|
| 90% | 9 out of 10 intervals contain true value | Exploratory analysis |
| 95% | 19 out of 20 intervals contain true value | Standard reporting |
| 99% | 99 out of 100 intervals contain true value | High-stakes decisions |

### Example Usage

```python
# Calculate bootstrap confidence intervals
intervals = stats.bootstrap_confidence_intervals(
    scores=model_a_scores,
    n_bootstrap=10000,
    confidence_levels=[0.95, 0.99],
    method='bca'
)

print(f"95% CI: {intervals[0.95]}")
print(f"99% CI: {intervals[0.99]}")
```

### Best Practices

- Use at least 1,000 bootstrap samples (10,000 for publication)
- BCa method preferred over basic/percentile methods
- Ensure original sample is representative
- Report both point estimate and confidence interval

## Effect Size Calculations

### Purpose
Effect sizes quantify the magnitude of differences between groups, providing practical significance beyond statistical significance. Essential for interpreting real-world impact of model improvements.

### Cohen's d (Standardized Mean Difference)

**Formula:**
```
d = (μ₁ - μ₂) / σ_pooled
```

Where σ_pooled is the pooled standard deviation.

**Interpretation:**
| Cohen's d | Effect Size | Practical Meaning |
|-----------|-------------|-------------------|
| 0.0 - 0.2 | Negligible | No meaningful difference |
| 0.2 - 0.5 | Small | Minor improvement |
| 0.5 - 0.8 | Medium | Moderate improvement |
| > 0.8 | Large | Substantial improvement |

### Hedge's g (Small Sample Correction)

Corrects Cohen's d for small sample bias:
```
g = d × J
where J = 1 - 3/(4(n₁ + n₂) - 9)
```

Use when total sample size < 20.

### Glass's Δ (Control Group Standard Deviation)

```
Δ = (μ₁ - μ₂) / σ_control
```

Use when one group (usually baseline model) serves as control with different variance.

### Probability of Superiority (A₁₂)

Probability that a randomly chosen score from Model A exceeds a randomly chosen score from Model B.

**Interpretation:**
- 0.5 = Models equivalent
- 0.56+ = Model A probably superior
- 0.64+ = Model A clearly superior

### Example Usage

```python
effect_sizes = calculate_effect_sizes(model_a_scores, model_b_scores)

print(f"Cohen's d: {effect_sizes['cohens_d']:.3f}")
print(f"Interpretation: {effect_sizes['interpretation']['cohens_d']}")
print(f"Probability of superiority: {effect_sizes['probability_superiority']:.3f}")
```

## Power Analysis

### Purpose
Power analysis determines the sample size needed to detect an effect of a given magnitude with specified probability. Critical for planning evaluations and interpreting non-significant results.

### Key Concepts

- **Statistical Power (1-β):** Probability of detecting a true effect
- **Type I Error (α):** False positive rate (typically 0.05)
- **Type II Error (β):** False negative rate
- **Effect Size:** Magnitude of difference to detect

### Sample Size Calculation

For a given effect size δ, significance level α, and desired power:

```python
power_result = power_analysis(
    effect_size=0.5,  # Medium effect
    alpha=0.05,       # 5% Type I error
    power=0.80        # 80% power
)

required_n = power_result['required_sample_size']
```

### Power Guidelines

| Power | Interpretation | Use Case |
|--------|----------------|----------|
| 0.50 | Coin flip | Insufficient |
| 0.70 | Moderate | Exploratory |
| 0.80 | Good | Standard |
| 0.90 | High | Critical applications |
| 0.95 | Very high | High-stakes decisions |

### Post-hoc Power Analysis

After conducting a study, calculate achieved power:
```python
# Calculate power for observed effect size and sample size
observed_power = ttest_power(
    effect_size=observed_cohens_d,
    nobs=actual_sample_size,
    alpha=0.05
)
```

## Multiple Testing Corrections

### Purpose
When conducting multiple statistical tests, the probability of at least one false positive increases. Multiple testing corrections control error rates across the family of tests.

### Family-Wise Error Rate (FWER) Methods

#### Bonferroni Correction
**Most conservative:** Multiply each p-value by number of tests
```
p_adjusted = min(p_original × m, 1)
```

**Use when:** False positives must be minimized at all costs

#### Holm-Bonferroni Method
**Step-down procedure:** More powerful than Bonferroni
1. Sort p-values ascending
2. For i-th smallest p-value: p_adj = p × (m - i + 1)
3. Maintain monotonicity

**Use when:** Balancing power and FWER control

### False Discovery Rate (FDR) Methods

#### Benjamini-Hochberg (BH) Procedure
**Controls FDR:** Expected proportion of false discoveries among rejections
```
For i-th sorted p-value: p_adj = p × m / i
```

**Use when:** Some false positives acceptable, want more power

#### Benjamini-Yekutieli (BY) Procedure
**For dependent tests:** More conservative than BH
```
p_adj = p × m × c(m) / i
where c(m) = Σ(1/j) for j=1 to m
```

### Method Selection Guide

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Exploratory analysis | FDR (BH) | Balance power and false discoveries |
| Confirmatory studies | FWER (Holm) | Strict error control |
| Safety-critical | FWER (Bonferroni) | Minimize any false positives |
| Many correlated tests | FDR (BY) | Accounts for dependence |

### Example Usage

```python
# Multiple model comparisons
p_values = [0.001, 0.05, 0.02, 0.10, 0.001]

# Apply Benjamini-Hochberg FDR correction
fdr_results = multiple_testing_correction(
    p_values=p_values,
    method='fdr_bh',
    alpha=0.05
)

print(f"Rejected: {fdr_results['rejected_hypotheses']} out of {fdr_results['n_tests']}")
```

## Integration Examples

### Complete Model Comparison Workflow

```python
import numpy as np
from src.evaluation.statistical_significance import (
    StatisticalSignificance, calculate_effect_sizes,
    power_analysis, multiple_testing_correction
)

# Initialize statistical engine
stats = StatisticalSignificance(random_state=42)

# Model performance data
model_a_scores = [0.85, 0.87, 0.83, 0.89, 0.86, 0.84, 0.88]
model_b_scores = [0.78, 0.80, 0.76, 0.82, 0.79, 0.77, 0.81]

# Step 1: Statistical significance
t_result = stats.welch_t_test(model_a_scores, model_b_scores)
print(f"Statistical test: {t_result.interpretation}")

# Step 2: Effect size
effects = calculate_effect_sizes(model_a_scores, model_b_scores)
print(f"Effect size (Cohen's d): {effects['cohens_d']:.3f} ({effects['interpretation']['cohens_d']})")

# Step 3: Confidence intervals
ci_diff = t_result.confidence_interval
print(f"95% CI for difference: ({ci_diff[0]:.3f}, {ci_diff[1]:.3f})")

# Step 4: Power analysis
power_result = power_analysis(effect_size=effects['cohens_d'])
print(f"Required sample size for replication: {power_result['required_sample_size']} per group")

# Step 5: Bootstrap confidence intervals for robustness
bootstrap_ci = stats.bootstrap_confidence_intervals(model_a_scores)
print(f"Bootstrap 95% CI for Model A: {bootstrap_ci[0.95]}")
```

### Multiple Model Comparison

```python
# Comparing multiple models requires multiple testing correction
models = {
    'baseline': [0.75, 0.77, 0.73, 0.79, 0.76],
    'model_v1': [0.82, 0.84, 0.80, 0.86, 0.83],
    'model_v2': [0.85, 0.87, 0.83, 0.89, 0.86],
    'model_v3': [0.81, 0.83, 0.79, 0.85, 0.82]
}

# Pairwise comparisons
comparisons = []
p_values = []

for name_a, scores_a in models.items():
    for name_b, scores_b in models.items():
        if name_a < name_b:  # Avoid duplicate comparisons
            result = stats.welch_t_test(scores_a, scores_b)
            comparisons.append(f"{name_a} vs {name_b}")
            p_values.append(result.p_value)

# Apply FDR correction
corrected = multiple_testing_correction(p_values, method='fdr_bh')

# Report results
for i, comparison in enumerate(comparisons):
    original_p = p_values[i]
    adjusted_p = corrected['adjusted_p_values'][i]
    significant = corrected['significant'][i]

    print(f"{comparison}: p={original_p:.4f}, p_adj={adjusted_p:.4f}, significant={significant}")
```

## Best Practices Summary

### Statistical Testing
1. **Always check assumptions** before applying tests
2. **Report effect sizes** alongside p-values
3. **Use appropriate corrections** for multiple comparisons
4. **Consider practical significance** not just statistical significance

### Sample Size Planning
1. **Conduct power analysis** before data collection
2. **Plan for 80%+ power** to detect meaningful effects
3. **Consider effect sizes** from similar studies
4. **Account for attrition** in longitudinal studies

### Reporting Standards
1. **Report exact p-values** (not just "p < 0.05")
2. **Include confidence intervals** for all estimates
3. **Describe statistical methods** used
4. **Acknowledge limitations** and assumptions

### Cambodia-Specific Considerations

When evaluating AI models for Cambodia's economy:

1. **Cultural Context:** Consider Khmer language nuances in performance metrics
2. **Economic Impact:** Weight effect sizes by economic sector importance
3. **Practical Significance:** Small statistical effects may have large economic impact
4. **Reproducibility:** Use robust methods given limited validation data
5. **Stakeholder Communication:** Translate statistical results to business impact

## References

1. **Welch, B.L. (1947).** "The generalization of 'Student's' problem when several different population variances are involved." *Biometrika*, 34(1-2), 28-35.

2. **Efron, B., & Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap.* Chapman & Hall/CRC.

3. **Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences.* Lawrence Erlbaum Associates.

4. **Benjamini, Y., & Hochberg, Y. (1995).** "Controlling the false discovery rate: A practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society*, 57(1), 289-300.

5. **Cumming, G. (2014).** "The new statistics: Why and how." *Psychological Science*, 25(1), 7-29.

6. **Wasserstein, R. L., & Lazar, N. A. (2016).** "The ASA statement on p-values: Context, process, and purpose." *The American Statistician*, 70(2), 129-133.

7. **Lakens, D. (2013).** "Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs." *Frontiers in Psychology*, 4, 863.

8. **Holm, S. (1979).** "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.

## Appendix: Quick Reference

### Function Quick Reference

```python
# Statistical significance testing
result = stats.welch_t_test(model_a, model_b)

# Effect sizes
effects = calculate_effect_sizes(model_a, model_b)

# Bootstrap confidence intervals
ci = stats.bootstrap_confidence_intervals(scores, n_bootstrap=10000)

# Power analysis
power = power_analysis(effect_size=0.5, power=0.8)

# Multiple testing correction
corrected = multiple_testing_correction(p_values, method='fdr_bh')
```

### Interpretation Templates

**Statistical Significance:**
> "Model A significantly outperformed Model B (Welch's t-test: t = X.XX, df = XX.X, p = 0.XXX, 95% CI: [X.XX, X.XX])."

**Effect Size:**
> "The effect size was large (Cohen's d = X.XX), indicating a practically meaningful difference between models."

**Multiple Comparisons:**
> "After applying Benjamini-Hochberg FDR correction, X out of Y comparisons remained significant (q < 0.05)."

---

*This documentation provides the statistical foundation for rigorous AI model evaluation in the GDP-eval framework. For specific implementation details, refer to the source code and test suite.*