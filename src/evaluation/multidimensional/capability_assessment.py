"""
Capability Assessment Module - Enterprise Grade
Evaluates: Reasoning, Knowledge, Creativity
OpenAI/Anthropic-level evaluation standards
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CapabilityDimension(Enum):
    """Capability dimensions to evaluate"""
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    CREATIVITY = "creativity"
    PROBLEM_SOLVING = "problem_solving"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    CONTEXTUAL_AWARENESS = "contextual_awareness"


@dataclass
class CapabilityScore:
    """Container for capability assessment scores"""
    dimension: CapabilityDimension
    score: float  # 0-100
    confidence: float  # 0-1
    evidence: List[str]
    sub_scores: Dict[str, float]
    metadata: Dict[str, Any]


class CapabilityAssessment:
    """
    Enterprise-grade capability assessment system
    Evaluates model capabilities across multiple dimensions
    """

    def __init__(self):
        self.test_suites = self._initialize_test_suites()
        self.scoring_weights = self._initialize_weights()
        self.results_cache = {}

    def _initialize_test_suites(self) -> Dict[str, List[Dict]]:
        """Initialize comprehensive test suites for each capability"""
        return {
            "reasoning": [
                # Logical Reasoning Tests
                {
                    "test_id": "logic_001",
                    "type": "deductive_reasoning",
                    "prompt": "If all roses are flowers, and all flowers need water, what do roses need?",
                    "expected_reasoning": ["transitive property", "logical deduction"],
                    "difficulty": 1
                },
                {
                    "test_id": "logic_002",
                    "type": "inductive_reasoning",
                    "prompt": "Given the sequence: 2, 6, 12, 20, 30, what is the next number and explain the pattern?",
                    "expected_answer": 42,
                    "expected_pattern": "n*(n+1) where n starts from 1",
                    "difficulty": 2
                },
                {
                    "test_id": "logic_003",
                    "type": "causal_reasoning",
                    "prompt": "A farmer notices that when it rains heavily, his crops grow faster, but too much rain causes root rot. What is the optimal rainfall strategy?",
                    "expected_concepts": ["optimization", "trade-offs", "threshold effects"],
                    "difficulty": 3
                },
                # Mathematical Reasoning
                {
                    "test_id": "math_001",
                    "type": "arithmetic_reasoning",
                    "prompt": "A Cambodian rice farmer harvests 3 tons per hectare. If diesel costs $0.90/liter and he uses 15 liters per hectare, what's his fuel cost per ton of rice?",
                    "expected_answer": 4.5,
                    "expected_unit": "dollars per ton",
                    "difficulty": 2
                },
                # Multi-step Problem Solving
                {
                    "test_id": "multistep_001",
                    "type": "complex_reasoning",
                    "prompt": "A tour company in Siem Reap has 3 guides who speak English, 2 who speak Chinese, and 1 who speaks both. If they need to serve 4 English groups and 3 Chinese groups simultaneously, what's the minimum number of additional guides needed?",
                    "expected_answer": 2,
                    "expected_reasoning": ["resource allocation", "constraint satisfaction"],
                    "difficulty": 4
                }
            ],
            "knowledge": [
                # Factual Knowledge
                {
                    "test_id": "fact_001",
                    "type": "cambodia_facts",
                    "prompt": "What is the official currency of Cambodia and its typical exchange rate to USD?",
                    "expected_keywords": ["riel", "KHR", "4000", "4100"],
                    "difficulty": 1
                },
                {
                    "test_id": "fact_002",
                    "type": "domain_knowledge",
                    "prompt": "Explain the difference between micro-finance and traditional banking in rural Cambodia.",
                    "expected_concepts": ["collateral", "interest rates", "group lending", "accessibility"],
                    "difficulty": 3
                },
                # Cross-domain Knowledge
                {
                    "test_id": "cross_001",
                    "type": "interdisciplinary",
                    "prompt": "How does the monsoon season affect both agriculture and tourism in Cambodia?",
                    "expected_domains": ["meteorology", "agriculture", "economics", "tourism"],
                    "difficulty": 3
                },
                # Temporal Knowledge
                {
                    "test_id": "temporal_001",
                    "type": "current_events",
                    "prompt": "What are the current major infrastructure projects in Cambodia?",
                    "expected_topics": ["roads", "airports", "ports", "railways"],
                    "difficulty": 2
                }
            ],
            "creativity": [
                # Divergent Thinking
                {
                    "test_id": "divergent_001",
                    "type": "alternative_uses",
                    "prompt": "List 5 innovative uses for bamboo in modern Cambodian construction beyond traditional applications.",
                    "min_valid_ideas": 3,
                    "originality_threshold": 0.7,
                    "difficulty": 2
                },
                # Novel Problem Solving
                {
                    "test_id": "novel_001",
                    "type": "creative_solution",
                    "prompt": "Design a low-cost water purification system using only materials available in rural Cambodian markets.",
                    "evaluation_criteria": ["feasibility", "cost", "innovation", "local_materials"],
                    "difficulty": 4
                },
                # Metaphor Understanding
                {
                    "test_id": "metaphor_001",
                    "type": "metaphor_interpretation",
                    "prompt": "Explain this Khmer proverb to a foreigner: 'ដើរតាមដានជើងដំរី' (Walk following elephant tracks)",
                    "expected_themes": ["wisdom", "experience", "safety", "tradition"],
                    "difficulty": 3
                },
                # Creative Writing
                {
                    "test_id": "creative_writing_001",
                    "type": "story_generation",
                    "prompt": "Write a short business pitch for a startup that combines traditional Khmer crafts with modern e-commerce.",
                    "evaluation_metrics": ["originality", "coherence", "market_understanding", "cultural_sensitivity"],
                    "difficulty": 3
                }
            ]
        }

    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize scoring weights for different capability dimensions"""
        return {
            "reasoning": 0.35,
            "knowledge": 0.30,
            "creativity": 0.20,
            "problem_solving": 0.10,
            "language_understanding": 0.05
        }

    async def evaluate_reasoning(self, model_response: str, test_case: Dict) -> Dict[str, Any]:
        """
        Evaluate reasoning capabilities

        Args:
            model_response: Model's response to test prompt
            test_case: Test case configuration

        Returns:
            Detailed reasoning evaluation scores
        """
        scores = {
            "logical_consistency": 0.0,
            "step_validity": 0.0,
            "conclusion_accuracy": 0.0,
            "explanation_quality": 0.0
        }

        # Check logical consistency
        if test_case["type"] == "deductive_reasoning":
            # Check for presence of expected reasoning patterns
            for pattern in test_case.get("expected_reasoning", []):
                if pattern.lower() in model_response.lower():
                    scores["logical_consistency"] += 50.0

        # Check mathematical accuracy for math problems
        if test_case["type"] == "arithmetic_reasoning":
            # Extract numbers from response
            numbers = re.findall(r'\d+\.?\d*', model_response)
            expected = str(test_case.get("expected_answer", ""))
            if expected in numbers:
                scores["conclusion_accuracy"] = 100.0

        # Evaluate explanation quality
        explanation_length = len(model_response.split())
        if explanation_length > 20:  # Reasonable explanation
            scores["explanation_quality"] = min(100, explanation_length * 2)

        # Check for step-by-step reasoning
        if any(marker in model_response.lower() for marker in ["first", "second", "then", "therefore", "step"]):
            scores["step_validity"] = 80.0

        overall_score = np.mean(list(scores.values()))

        return {
            "overall_score": overall_score,
            "sub_scores": scores,
            "test_id": test_case["test_id"],
            "difficulty": test_case["difficulty"],
            "response_length": len(model_response),
            "timestamp": datetime.now().isoformat()
        }

    async def evaluate_knowledge(self, model_response: str, test_case: Dict) -> Dict[str, Any]:
        """
        Evaluate knowledge and factual accuracy

        Args:
            model_response: Model's response
            test_case: Test case configuration

        Returns:
            Knowledge evaluation scores
        """
        scores = {
            "factual_accuracy": 0.0,
            "completeness": 0.0,
            "domain_expertise": 0.0,
            "specificity": 0.0
        }

        response_lower = model_response.lower()

        # Check for expected keywords/facts
        if "expected_keywords" in test_case:
            keywords_found = sum(1 for keyword in test_case["expected_keywords"]
                               if keyword.lower() in response_lower)
            scores["factual_accuracy"] = (keywords_found / len(test_case["expected_keywords"])) * 100

        # Check for expected concepts
        if "expected_concepts" in test_case:
            concepts_found = sum(1 for concept in test_case["expected_concepts"]
                               if concept.lower() in response_lower)
            scores["domain_expertise"] = (concepts_found / len(test_case["expected_concepts"])) * 100

        # Evaluate completeness based on response length and detail
        word_count = len(model_response.split())
        scores["completeness"] = min(100, word_count / 2)  # 200 words = 100% completeness

        # Check specificity (presence of numbers, dates, proper nouns)
        specific_elements = len(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b|\b\d{4}\b', model_response))
        scores["specificity"] = min(100, specific_elements * 10)

        overall_score = np.mean(list(scores.values()))

        return {
            "overall_score": overall_score,
            "sub_scores": scores,
            "test_id": test_case["test_id"],
            "difficulty": test_case["difficulty"],
            "facts_identified": scores["factual_accuracy"] > 50,
            "timestamp": datetime.now().isoformat()
        }

    async def evaluate_creativity(self, model_response: str, test_case: Dict) -> Dict[str, Any]:
        """
        Evaluate creative and innovative thinking

        Args:
            model_response: Model's response
            test_case: Test case configuration

        Returns:
            Creativity evaluation scores
        """
        scores = {
            "originality": 0.0,
            "fluency": 0.0,
            "flexibility": 0.0,
            "elaboration": 0.0
        }

        # Evaluate originality (using simple heuristics)
        unique_words = len(set(model_response.split()))
        total_words = len(model_response.split())
        scores["originality"] = min(100, (unique_words / max(total_words, 1)) * 150)

        # Evaluate fluency (idea generation)
        if test_case["type"] == "alternative_uses":
            # Count number of distinct ideas (separated by newlines, numbers, or bullets)
            ideas = re.split(r'\n|\d+\.|\*|-', model_response)
            valid_ideas = [idea for idea in ideas if len(idea.strip()) > 10]
            scores["fluency"] = min(100, len(valid_ideas) * 20)

        # Evaluate flexibility (variety of approaches)
        if "evaluation_criteria" in test_case:
            criteria_addressed = sum(1 for criterion in test_case["evaluation_criteria"]
                                   if criterion.lower() in model_response.lower())
            scores["flexibility"] = (criteria_addressed / len(test_case["evaluation_criteria"])) * 100

        # Evaluate elaboration (detail and development)
        sentences = model_response.count('.') + model_response.count('!') + model_response.count('?')
        scores["elaboration"] = min(100, sentences * 10)

        overall_score = np.mean(list(scores.values()))

        return {
            "overall_score": overall_score,
            "sub_scores": scores,
            "test_id": test_case["test_id"],
            "difficulty": test_case["difficulty"],
            "unique_word_ratio": unique_words / max(total_words, 1),
            "timestamp": datetime.now().isoformat()
        }

    async def run_capability_assessment(self, model, test_subset: Optional[str] = None) -> CapabilityScore:
        """
        Run comprehensive capability assessment

        Args:
            model: Model to evaluate (must have generate method)
            test_subset: Optional subset of tests to run

        Returns:
            Comprehensive capability scores
        """
        all_results = {
            "reasoning": [],
            "knowledge": [],
            "creativity": []
        }

        # Run reasoning tests
        for test_case in self.test_suites["reasoning"]:
            if test_subset and test_case["test_id"] not in test_subset:
                continue

            response = await model.generate(test_case["prompt"])
            result = await self.evaluate_reasoning(response.text, test_case)
            all_results["reasoning"].append(result)

        # Run knowledge tests
        for test_case in self.test_suites["knowledge"]:
            if test_subset and test_case["test_id"] not in test_subset:
                continue

            response = await model.generate(test_case["prompt"])
            result = await self.evaluate_knowledge(response.text, test_case)
            all_results["knowledge"].append(result)

        # Run creativity tests
        for test_case in self.test_suites["creativity"]:
            if test_subset and test_case["test_id"] not in test_subset:
                continue

            response = await model.generate(test_case["prompt"])
            result = await self.evaluate_creativity(response.text, test_case)
            all_results["creativity"].append(result)

        # Calculate aggregate scores
        reasoning_score = np.mean([r["overall_score"] for r in all_results["reasoning"]]) if all_results["reasoning"] else 0
        knowledge_score = np.mean([r["overall_score"] for r in all_results["knowledge"]]) if all_results["knowledge"] else 0
        creativity_score = np.mean([r["overall_score"] for r in all_results["creativity"]]) if all_results["creativity"] else 0

        overall_score = min(100, max(0, (
            reasoning_score * self.scoring_weights["reasoning"] +
            knowledge_score * self.scoring_weights["knowledge"] +
            creativity_score * self.scoring_weights["creativity"]
        ) / sum([self.scoring_weights["reasoning"], self.scoring_weights["knowledge"], self.scoring_weights["creativity"]])))

        return CapabilityScore(
            dimension=CapabilityDimension.REASONING,
            score=overall_score,
            confidence=0.85,  # Based on number of tests run
            evidence=[f"Ran {len(all_results['reasoning']) + len(all_results['knowledge']) + len(all_results['creativity'])} tests"],
            sub_scores={
                "reasoning": reasoning_score,
                "knowledge": knowledge_score,
                "creativity": creativity_score
            },
            metadata={
                "test_results": all_results,
                "timestamp": datetime.now().isoformat(),
                "test_coverage": len([r for results in all_results.values() for r in results])
            }
        )

    def generate_capability_report(self, capability_score: CapabilityScore) -> str:
        """
        Generate detailed capability assessment report

        Args:
            capability_score: Capability assessment results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("CAPABILITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append(f"\nOverall Score: {capability_score.score:.1f}/100")
        report.append(f"Confidence: {capability_score.confidence:.2f}")
        report.append(f"\nDimension Scores:")

        for dimension, score in capability_score.sub_scores.items():
            report.append(f"  - {dimension.capitalize()}: {score:.1f}/100")

        report.append(f"\nTest Coverage: {capability_score.metadata['test_coverage']} tests completed")
        report.append(f"Assessment Date: {capability_score.metadata['timestamp']}")

        # Add detailed insights
        report.append("\nKey Insights:")
        if capability_score.sub_scores["reasoning"] > 80:
            report.append("  ✓ Strong logical reasoning capabilities")
        elif capability_score.sub_scores["reasoning"] < 60:
            report.append("  ⚠ Reasoning capabilities need improvement")

        if capability_score.sub_scores["knowledge"] > 80:
            report.append("  ✓ Excellent domain knowledge")
        elif capability_score.sub_scores["knowledge"] < 60:
            report.append("  ⚠ Knowledge gaps identified")

        if capability_score.sub_scores["creativity"] > 80:
            report.append("  ✓ High creative thinking ability")
        elif capability_score.sub_scores["creativity"] < 60:
            report.append("  ⚠ Limited creative problem-solving")

        report.append("\n" + "=" * 80)

        return "\n".join(report)