"""
Consistency Analysis Module - Enterprise Grade
Evaluates: Temporal consistency, cross-prompt coherence, output stability
OpenAI/Anthropic-level evaluation standards
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConsistencyTestType(Enum):
    """Types of consistency tests"""
    TEMPORAL = "temporal"
    CROSS_PROMPT = "cross_prompt"
    SEMANTIC = "semantic"
    FACTUAL = "factual"
    STYLISTIC = "stylistic"
    BEHAVIORAL = "behavioral"


@dataclass
class ConsistencyScore:
    """Container for consistency analysis scores"""
    test_type: ConsistencyTestType
    consistency_score: float  # 0-100
    variance: float  # Statistical variance
    drift_detected: bool
    anomalies: List[str]
    patterns: List[str]
    metadata: Dict[str, Any]


class ConsistencyAnalysis:
    """
    Enterprise-grade consistency analysis system
    Evaluates model consistency across various dimensions
    """

    def __init__(self):
        self.test_prompts = self._initialize_test_prompts()
        self.consistency_thresholds = self._initialize_thresholds()
        self.response_cache = defaultdict(list)

    def _initialize_test_prompts(self) -> Dict[str, List[Dict]]:
        """Initialize consistency test prompts"""
        return {
            "temporal": [
                {
                    "prompt": "What is the capital of Cambodia?",
                    "expected_consistency": ["Phnom Penh"],
                    "category": "factual",
                    "repetitions": 5,
                    "delay_seconds": 1
                },
                {
                    "prompt": "Explain photosynthesis in simple terms",
                    "expected_concepts": ["light", "plants", "energy", "oxygen", "carbon dioxide"],
                    "category": "explanatory",
                    "repetitions": 3,
                    "delay_seconds": 2
                }
            ],
            "cross_prompt": [
                {
                    "prompts": [
                        "What is 15% of 200?",
                        "Calculate 0.15 × 200",
                        "If 200 is reduced by 85%, what remains?"
                    ],
                    "expected_answer": 30,
                    "category": "mathematical"
                },
                {
                    "prompts": [
                        "Describe the benefits of exercise",
                        "Why should people exercise regularly?",
                        "What are the advantages of physical activity?"
                    ],
                    "expected_themes": ["health", "fitness", "wellbeing"],
                    "category": "thematic"
                }
            ],
            "semantic": [
                {
                    "base_prompt": "Explain the importance of clean water",
                    "paraphrases": [
                        "Why is clean water important?",
                        "Describe the significance of safe drinking water",
                        "What makes clean water essential?"
                    ],
                    "semantic_similarity_threshold": 0.75
                }
            ],
            "factual": [
                {
                    "fact_checks": [
                        "The Mekong River flows through Cambodia",
                        "Cambodia shares borders with Thailand, Laos, and Vietnam",
                        "Angkor Wat is located in Siem Reap province"
                    ],
                    "truth_values": [True, True, True],
                    "category": "geographical"
                },
                {
                    "fact_checks": [
                        "1 USD equals approximately 4,100 Cambodian Riel",
                        "Cambodia's population is about 17 million",
                        "The official language of Cambodia is Khmer"
                    ],
                    "truth_values": [True, True, True],
                    "category": "demographic"
                }
            ],
            "stylistic": [
                {
                    "style_prompt": "Write a formal business email about a meeting",
                    "style": "formal",
                    "indicators": ["Dear", "Sincerely", "would like to", "please"]
                },
                {
                    "style_prompt": "Explain quantum physics to a 5-year-old",
                    "style": "simple",
                    "indicators": ["like", "imagine", "think of", "just like"]
                }
            ]
        }

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize consistency thresholds"""
        return {
            "temporal_consistency": 0.85,  # 85% similarity required
            "cross_prompt_consistency": 0.80,
            "semantic_consistency": 0.75,
            "factual_accuracy": 0.95,
            "stylistic_consistency": 0.70,
            "variance_threshold": 0.15,  # Max acceptable variance
            "drift_threshold": 0.20  # Max acceptable drift
        }

    async def test_temporal_consistency(self, model) -> Dict[str, Any]:
        """
        Test consistency of responses over time

        Args:
            model: Model to test

        Returns:
            Temporal consistency results
        """
        results = {
            "consistency_scores": [],
            "response_variations": [],
            "drift_detected": False,
            "temporal_stability": 0.0
        }

        for test_case in self.test_prompts["temporal"]:
            responses = []
            response_hashes = []

            # Collect multiple responses over time
            for i in range(test_case["repetitions"]):
                try:
                    response = await model.generate(test_case["prompt"])
                    responses.append(response.text)
                    response_hashes.append(hashlib.md5(response.text.encode()).hexdigest())

                    # Add delay between requests
                    if i < test_case["repetitions"] - 1:
                        await asyncio.sleep(test_case["delay_seconds"])
                except Exception as e:
                    logger.warning(f"Temporal test failed: {e}")
                    responses.append("")

            # Analyze consistency
            if responses:
                # Check exact matches
                unique_responses = len(set(response_hashes))
                exact_consistency = 1.0 / unique_responses if unique_responses > 0 else 0

                # Check semantic consistency
                semantic_consistency = self._calculate_semantic_consistency(responses)

                # Check for expected elements
                if "expected_consistency" in test_case:
                    expected_present = all(
                        any(exp.lower() in resp.lower()
                            for resp in responses)
                        for exp in test_case["expected_consistency"]
                    )
                else:
                    expected_present = True

                consistency_score = (exact_consistency * 0.3 +
                                   semantic_consistency * 0.5 +
                                   (1.0 if expected_present else 0.0) * 0.2)

                results["consistency_scores"].append({
                    "prompt": test_case["prompt"][:50] + "...",
                    "score": consistency_score * 100,
                    "unique_responses": unique_responses,
                    "total_responses": len(responses)
                })

                # Detect drift (responses changing over time)
                if len(responses) >= 3:
                    early_response = responses[0]
                    late_response = responses[-1]
                    drift = 1.0 - self._calculate_similarity(early_response, late_response)
                    if drift > self.consistency_thresholds["drift_threshold"]:
                        results["drift_detected"] = True

        # Calculate overall temporal stability
        if results["consistency_scores"]:
            results["temporal_stability"] = np.mean([s["score"] for s in results["consistency_scores"]])

        return results

    async def test_cross_prompt_consistency(self, model) -> Dict[str, Any]:
        """
        Test consistency across different phrasings of same question

        Args:
            model: Model to test

        Returns:
            Cross-prompt consistency results
        """
        results = {
            "consistency_scores": [],
            "inconsistent_answers": [],
            "cross_prompt_coherence": 0.0
        }

        for test_case in self.test_prompts["cross_prompt"]:
            responses = []

            # Get responses for all prompt variations
            for prompt in test_case["prompts"]:
                try:
                    response = await model.generate(prompt)
                    responses.append(response.text)
                except Exception as e:
                    logger.warning(f"Cross-prompt test failed: {e}")
                    responses.append("")

            # Analyze consistency
            if all(responses):
                # Check mathematical consistency
                if test_case["category"] == "mathematical":
                    extracted_numbers = []
                    for resp in responses:
                        numbers = self._extract_numbers(resp)
                        if numbers:
                            extracted_numbers.extend(numbers)

                    expected = test_case.get("expected_answer")
                    if expected:
                        consistency = sum(1 for n in extracted_numbers
                                        if abs(n - expected) < 0.01) / len(responses)
                    else:
                        # Check if all extracted numbers are similar
                        if extracted_numbers:
                            consistency = 1.0 - (np.std(extracted_numbers) / (np.mean(extracted_numbers) + 1e-6))
                        else:
                            consistency = 0.0

                # Check thematic consistency
                elif test_case["category"] == "thematic":
                    theme_presence = []
                    for theme in test_case.get("expected_themes", []):
                        present_count = sum(1 for resp in responses
                                          if theme.lower() in resp.lower())
                        theme_presence.append(present_count / len(responses))

                    consistency = np.mean(theme_presence) if theme_presence else 0.0

                else:
                    # General semantic consistency
                    consistency = self._calculate_semantic_consistency(responses)

                results["consistency_scores"].append({
                    "category": test_case["category"],
                    "score": consistency * 100,
                    "prompt_count": len(test_case["prompts"])
                })

                if consistency < self.consistency_thresholds["cross_prompt_consistency"]:
                    results["inconsistent_answers"].append({
                        "category": test_case["category"],
                        "consistency": consistency,
                        "issue": "Low cross-prompt consistency"
                    })

        # Calculate overall coherence
        if results["consistency_scores"]:
            results["cross_prompt_coherence"] = np.mean([s["score"] for s in results["consistency_scores"]])

        return results

    async def test_factual_consistency(self, model) -> Dict[str, Any]:
        """
        Test consistency of factual claims

        Args:
            model: Model to test

        Returns:
            Factual consistency results
        """
        results = {
            "factual_accuracy": 0.0,
            "correct_facts": [],
            "incorrect_facts": [],
            "consistency_violations": []
        }

        correct_count = 0
        total_count = 0

        for test_case in self.test_prompts["factual"]:
            for fact, truth_value in zip(test_case["fact_checks"], test_case["truth_values"]):
                total_count += 1

                # Test if model agrees with fact
                prompt = f"Is the following statement true or false? Answer with just 'true' or 'false': {fact}"

                try:
                    response = await model.generate(prompt)
                    response_lower = response.text.lower().strip()

                    # Extract boolean response
                    is_true_response = "true" in response_lower and "false" not in response_lower
                    is_false_response = "false" in response_lower and "true" not in response_lower

                    if is_true_response:
                        model_verdict = True
                    elif is_false_response:
                        model_verdict = False
                    else:
                        model_verdict = None  # Unclear response

                    # Check if model is correct
                    if model_verdict == truth_value:
                        correct_count += 1
                        results["correct_facts"].append({
                            "fact": fact[:50] + "...",
                            "category": test_case["category"]
                        })
                    else:
                        results["incorrect_facts"].append({
                            "fact": fact[:50] + "...",
                            "expected": truth_value,
                            "model_said": model_verdict,
                            "category": test_case["category"]
                        })

                    # Test consistency by asking same fact differently
                    rephrased = f"Please confirm whether this is accurate: {fact}"
                    response2 = await model.generate(rephrased)

                    # Check if responses are consistent
                    is_true_response2 = "true" in response2.text.lower() or "accurate" in response2.text.lower() or "correct" in response2.text.lower()

                    if (is_true_response != is_true_response2) and model_verdict is not None:
                        results["consistency_violations"].append({
                            "fact": fact[:50] + "...",
                            "issue": "Inconsistent factual response"
                        })

                except Exception as e:
                    logger.warning(f"Factual consistency test failed: {e}")

        results["factual_accuracy"] = (correct_count / max(total_count, 1)) * 100

        return results

    async def test_stylistic_consistency(self, model) -> Dict[str, Any]:
        """
        Test consistency in writing style

        Args:
            model: Model to test

        Returns:
            Stylistic consistency results
        """
        results = {
            "style_consistency": 0.0,
            "style_adherence": [],
            "style_violations": []
        }

        for test_case in self.test_prompts["stylistic"]:
            # Get multiple responses for same style
            responses = []
            for i in range(3):
                try:
                    modified_prompt = f"{test_case['style_prompt']} (Attempt {i+1})"
                    response = await model.generate(modified_prompt)
                    responses.append(response.text)
                except Exception as e:
                    logger.warning(f"Stylistic test failed: {e}")

            if responses:
                # Check for style indicators
                indicator_presence = []
                for response in responses:
                    response_lower = response.lower()
                    indicators_found = sum(1 for indicator in test_case["indicators"]
                                         if indicator.lower() in response_lower)
                    indicator_ratio = indicators_found / len(test_case["indicators"])
                    indicator_presence.append(indicator_ratio)

                # Calculate consistency of style
                style_consistency = 1.0 - np.std(indicator_presence) if len(indicator_presence) > 1 else 1.0
                mean_adherence = np.mean(indicator_presence)

                results["style_adherence"].append({
                    "style": test_case["style"],
                    "consistency": style_consistency * 100,
                    "adherence": mean_adherence * 100
                })

                if style_consistency < self.consistency_thresholds["stylistic_consistency"]:
                    results["style_violations"].append({
                        "style": test_case["style"],
                        "issue": "Inconsistent style application"
                    })

        # Calculate overall style consistency
        if results["style_adherence"]:
            results["style_consistency"] = np.mean([s["consistency"] for s in results["style_adherence"]])

        return results

    def _calculate_semantic_consistency(self, responses: List[str]) -> float:
        """Calculate semantic consistency across responses"""
        if len(responses) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._calculate_similarity(responses[i], responses[j])
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        import re
        numbers = []

        # Find all numbers in text
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                num = float(match)
                if -1e6 < num < 1e6:  # Reasonable range
                    numbers.append(num)
            except ValueError:
                continue

        return numbers

    async def run_consistency_analysis(self, model) -> ConsistencyScore:
        """
        Run complete consistency analysis suite

        Args:
            model: Model to evaluate

        Returns:
            Comprehensive consistency results
        """
        all_results = {}

        # Run temporal consistency tests
        logger.info("Testing temporal consistency...")
        temporal_results = await self.test_temporal_consistency(model)
        all_results["temporal"] = temporal_results

        # Run cross-prompt consistency tests
        logger.info("Testing cross-prompt consistency...")
        cross_prompt_results = await self.test_cross_prompt_consistency(model)
        all_results["cross_prompt"] = cross_prompt_results

        # Run factual consistency tests
        logger.info("Testing factual consistency...")
        factual_results = await self.test_factual_consistency(model)
        all_results["factual"] = factual_results

        # Run stylistic consistency tests
        logger.info("Testing stylistic consistency...")
        stylistic_results = await self.test_stylistic_consistency(model)
        all_results["stylistic"] = stylistic_results

        # Calculate overall consistency score
        consistency_scores = [
            temporal_results.get("temporal_stability", 0),
            cross_prompt_results.get("cross_prompt_coherence", 0),
            factual_results.get("factual_accuracy", 0),
            stylistic_results.get("style_consistency", 0)
        ]

        overall_consistency = min(100, max(0, np.mean(consistency_scores)))
        variance = np.var(consistency_scores)

        # Detect drift
        drift_detected = (
            temporal_results.get("drift_detected", False) or
            variance > self.consistency_thresholds["variance_threshold"] * 100
        )

        # Identify anomalies
        anomalies = []
        if temporal_results.get("drift_detected"):
            anomalies.append("Temporal drift detected")
        if cross_prompt_results.get("inconsistent_answers"):
            anomalies.append(f"{len(cross_prompt_results['inconsistent_answers'])} cross-prompt inconsistencies")
        if factual_results.get("incorrect_facts"):
            anomalies.append(f"{len(factual_results['incorrect_facts'])} factual errors")
        if stylistic_results.get("style_violations"):
            anomalies.append(f"{len(stylistic_results['style_violations'])} style violations")

        # Identify patterns
        patterns = []
        if overall_consistency > 85:
            patterns.append("High overall consistency")
        if temporal_results.get("temporal_stability", 0) > 90:
            patterns.append("Excellent temporal stability")
        if factual_results.get("factual_accuracy", 0) > 95:
            patterns.append("Strong factual reliability")

        return ConsistencyScore(
            test_type=ConsistencyTestType.TEMPORAL,
            consistency_score=overall_consistency,
            variance=variance,
            drift_detected=drift_detected,
            anomalies=anomalies[:5],
            patterns=patterns[:3],
            metadata={
                "test_results": all_results,
                "timestamp": datetime.now().isoformat(),
                "total_tests": sum([
                    len(temporal_results.get("consistency_scores", [])),
                    len(cross_prompt_results.get("consistency_scores", [])),
                    len(factual_results.get("correct_facts", [])) + len(factual_results.get("incorrect_facts", [])),
                    len(stylistic_results.get("style_adherence", []))
                ])
            }
        )

    def generate_consistency_report(self, consistency_score: ConsistencyScore) -> str:
        """
        Generate detailed consistency analysis report

        Args:
            consistency_score: Consistency analysis results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("CONSISTENCY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nOverall Consistency Score: {consistency_score.consistency_score:.1f}/100")
        report.append(f"Response Variance: {consistency_score.variance:.3f}")
        report.append(f"Drift Detected: {'Yes' if consistency_score.drift_detected else 'No'}")

        # Test breakdown
        test_results = consistency_score.metadata["test_results"]

        report.append("\nConsistency Dimensions:")
        report.append("-" * 40)

        if "temporal" in test_results:
            temp = test_results["temporal"]
            report.append(f"Temporal Consistency:")
            report.append(f"  - Stability Score: {temp.get('temporal_stability', 0):.1f}%")
            report.append(f"  - Drift Status: {'Detected' if temp.get('drift_detected', False) else 'None'}")

        if "cross_prompt" in test_results:
            cross = test_results["cross_prompt"]
            report.append(f"Cross-Prompt Consistency:")
            report.append(f"  - Coherence Score: {cross.get('cross_prompt_coherence', 0):.1f}%")
            report.append(f"  - Inconsistencies: {len(cross.get('inconsistent_answers', []))}")

        if "factual" in test_results:
            fact = test_results["factual"]
            report.append(f"Factual Consistency:")
            report.append(f"  - Accuracy: {fact.get('factual_accuracy', 0):.1f}%")
            report.append(f"  - Correct Facts: {len(fact.get('correct_facts', []))}")
            report.append(f"  - Errors: {len(fact.get('incorrect_facts', []))}")

        if "stylistic" in test_results:
            style = test_results["stylistic"]
            report.append(f"Stylistic Consistency:")
            report.append(f"  - Style Score: {style.get('style_consistency', 0):.1f}%")
            report.append(f"  - Violations: {len(style.get('style_violations', []))}")

        # Anomalies
        if consistency_score.anomalies:
            report.append("\nDetected Anomalies:")
            report.append("-" * 40)
            for anomaly in consistency_score.anomalies:
                report.append(f"  ⚠ {anomaly}")

        # Patterns
        if consistency_score.patterns:
            report.append("\nIdentified Patterns:")
            report.append("-" * 40)
            for pattern in consistency_score.patterns:
                report.append(f"  ✓ {pattern}")

        # Consistency assessment
        report.append("\nConsistency Assessment:")
        report.append("-" * 40)

        if consistency_score.consistency_score >= 90:
            report.append("  ✓ EXCELLENT - Model demonstrates exceptional consistency")
        elif consistency_score.consistency_score >= 80:
            report.append("  ✓ GOOD - Model shows strong consistency")
        elif consistency_score.consistency_score >= 70:
            report.append("  ⚠ FAIR - Some consistency issues detected")
        else:
            report.append("  ❌ POOR - Significant consistency problems")

        # Recommendations
        report.append("\nRecommendations:")
        report.append("-" * 40)

        if consistency_score.drift_detected:
            report.append("  1. Investigate temporal drift causes")
        if consistency_score.variance > 10:
            report.append("  2. Reduce response variance through targeted training")
        if test_results.get("factual", {}).get("incorrect_facts"):
            report.append("  3. Improve factual grounding and verification")
        if test_results.get("cross_prompt", {}).get("inconsistent_answers"):
            report.append("  4. Enhance semantic understanding for consistent responses")

        report.append("\n" + "=" * 80)

        return "\n".join(report)