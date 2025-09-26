"""
Grading System for GDPval Framework
Handles automated and human grading of model responses
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

from src.core.task_loader import EvaluationTask
from src.core.evaluator import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """Result of grading a model response"""
    task_id: str
    model_name: str
    total_score: float
    max_score: float
    percentage: float
    criteria_scores: Dict[str, float]
    feedback: Dict[str, str]
    grader_type: str
    timestamp: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert grading result to dictionary"""
        return asdict(self)

    def passed(self, threshold: float = 70.0) -> bool:
        """Check if the grade passes a threshold"""
        return self.percentage >= threshold


class BaseGrader:
    """Base class for grading systems"""

    def __init__(self, grader_name: str = "base_grader"):
        """
        Initialize grader

        Args:
            grader_name: Name of the grading system
        """
        self.grader_name = grader_name
        self.grading_results: List[GradingResult] = []

    def grade_response(
        self,
        task: EvaluationTask,
        result: EvaluationResult
    ) -> GradingResult:
        """
        Grade a model response for a task

        Args:
            task: The evaluation task
            result: The model's response

        Returns:
            Grading result
        """
        criteria_scores = {}
        feedback = {}
        max_score = 0

        # Grade each criterion
        for criterion in task.grading_criteria:
            criterion_id = criterion.get('criterion_id', 'unknown')
            criterion_name = criterion.get('criterion_name', 'Unknown')
            weight = criterion.get('weight', 1.0)
            max_points = criterion.get('max_score', 10.0)

            # Calculate score for this criterion
            score, criterion_feedback = self._grade_criterion(
                criterion,
                task,
                result.response
            )

            criteria_scores[criterion_id] = score * weight
            feedback[criterion_id] = criterion_feedback
            max_score += max_points * weight

        # Calculate total score
        total_score = sum(criteria_scores.values())
        percentage = (total_score / max_score * 100) if max_score > 0 else 0

        grading_result = GradingResult(
            task_id=task.task_id,
            model_name=result.model_name,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            criteria_scores=criteria_scores,
            feedback=feedback,
            grader_type=self.grader_name,
            timestamp=datetime.now().isoformat(),
            metadata={
                "category": task.category,
                "occupation": task.occupation,
                "response_length": len(result.response),
                "evaluation_latency": result.latency_ms
            }
        )

        self.grading_results.append(grading_result)
        return grading_result

    def _grade_criterion(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str
    ) -> Tuple[float, str]:
        """
        Grade a single criterion (to be implemented by subclasses)

        Args:
            criterion: Grading criterion
            task: Evaluation task
            response: Model response

        Returns:
            Tuple of (score, feedback)
        """
        raise NotImplementedError("Subclasses must implement _grade_criterion")

    def grade_batch(
        self,
        tasks: List[EvaluationTask],
        results: List[EvaluationResult]
    ) -> List[GradingResult]:
        """
        Grade multiple responses

        Args:
            tasks: List of tasks
            results: List of model responses

        Returns:
            List of grading results
        """
        grading_results = []

        # Create task index for quick lookup
        task_index = {task.task_id: task for task in tasks}

        for result in results:
            if result.task_id in task_index:
                task = task_index[result.task_id]
                grading_result = self.grade_response(task, result)
                grading_results.append(grading_result)
            else:
                logger.warning(f"No task found for result {result.task_id}")

        return grading_results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get grading statistics

        Returns:
            Dictionary with grading metrics
        """
        if not self.grading_results:
            return {"total_graded": 0}

        passing_threshold = 70.0
        passing_results = [r for r in self.grading_results if r.percentage >= passing_threshold]

        return {
            "total_graded": len(self.grading_results),
            "average_score": sum(r.percentage for r in self.grading_results) / len(self.grading_results),
            "passing_rate": len(passing_results) / len(self.grading_results) * 100,
            "highest_score": max(r.percentage for r in self.grading_results),
            "lowest_score": min(r.percentage for r in self.grading_results),
            "by_category": self._get_category_statistics()
        }

    def _get_category_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics broken down by category"""
        category_stats = {}

        for result in self.grading_results:
            category = result.metadata.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {
                    'count': 0,
                    'total_score': 0,
                    'max_score': 0,
                    'min_score': 100
                }

            stats = category_stats[category]
            stats['count'] += 1
            stats['total_score'] += result.percentage
            stats['max_score'] = max(stats['max_score'], result.percentage)
            stats['min_score'] = min(stats['min_score'], result.percentage)

        # Calculate averages
        for category, stats in category_stats.items():
            stats['average_score'] = stats['total_score'] / stats['count']
            del stats['total_score']  # Remove intermediate value

        return category_stats


class AutomatedGrader(BaseGrader):
    """Automated grading using rule-based and heuristic methods"""

    def __init__(self):
        """Initialize automated grader"""
        super().__init__("automated_grader")

    def _grade_criterion(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str
    ) -> Tuple[float, str]:
        """
        Grade a criterion using automated rules

        Args:
            criterion: Grading criterion
            task: Evaluation task
            response: Model response

        Returns:
            Tuple of (score, feedback)
        """
        criterion_id = criterion.get('criterion_id', '')
        max_score = criterion.get('max_score', 10.0)

        # Different grading strategies based on criterion type
        if criterion_id == 'accuracy':
            return self._grade_accuracy(criterion, task, response, max_score)
        elif criterion_id == 'completeness':
            return self._grade_completeness(criterion, task, response, max_score)
        elif criterion_id == 'language_quality':
            return self._grade_language_quality(criterion, task, response, max_score)
        elif criterion_id == 'technical_correctness':
            return self._grade_technical_correctness(criterion, task, response, max_score)
        else:
            # Default grading
            return self._grade_default(criterion, task, response, max_score)

    def _grade_accuracy(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str,
        max_score: float
    ) -> Tuple[float, str]:
        """Grade accuracy of the response"""
        score = max_score * 0.7  # Base score
        feedback_parts = []

        # Check for key terms from the task
        if 'expected_keywords' in criterion:
            keywords = criterion['expected_keywords']
            found_keywords = sum(1 for kw in keywords if kw.lower() in response.lower())
            keyword_ratio = found_keywords / len(keywords) if keywords else 0
            score = max_score * keyword_ratio
            feedback_parts.append(f"Found {found_keywords}/{len(keywords)} expected keywords")

        # Check numerical accuracy if applicable
        numbers_in_response = re.findall(r'\d+\.?\d*', response)
        if numbers_in_response:
            feedback_parts.append(f"Contains {len(numbers_in_response)} numerical values")
            score += max_score * 0.1

        feedback = "; ".join(feedback_parts) if feedback_parts else "Accuracy assessment completed"
        return score, feedback

    def _grade_completeness(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str,
        max_score: float
    ) -> Tuple[float, str]:
        """Grade completeness of the response"""
        score = 0
        feedback_parts = []

        # Check if expected sections are present
        if 'expected_sections' in task.prompt:
            sections = task.prompt['expected_sections']
            found_sections = 0
            for section in sections:
                # Check both Khmer and English section names
                if section in response or any(part in response for part in section.split('/')):
                    found_sections += 1

            section_ratio = found_sections / len(sections) if sections else 0
            score = max_score * section_ratio
            feedback_parts.append(f"Found {found_sections}/{len(sections)} expected sections")

        # Check response length
        min_length = criterion.get('min_response_length', 100)
        if len(response) >= min_length:
            score = max(score, max_score * 0.5)
            feedback_parts.append(f"Response length adequate ({len(response)} chars)")
        else:
            feedback_parts.append(f"Response too short ({len(response)}/{min_length} chars)")

        feedback = "; ".join(feedback_parts) if feedback_parts else "Completeness check done"
        return score, feedback

    def _grade_language_quality(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str,
        max_score: float
    ) -> Tuple[float, str]:
        """Grade language quality of the response"""
        score = max_score * 0.6  # Base score
        feedback_parts = []

        # Check for Khmer content if required
        if task.metadata.get('language') == 'khmer':
            khmer_chars = sum(1 for char in response if 0x1780 <= ord(char) <= 0x17FF)
            if khmer_chars > 0:
                khmer_ratio = khmer_chars / len(response)
                score = max_score * min(khmer_ratio * 2, 1.0)  # Bonus for Khmer content
                feedback_parts.append(f"Contains {khmer_ratio*100:.1f}% Khmer content")
            else:
                score = max_score * 0.3
                feedback_parts.append("No Khmer content found")

        # Check sentence structure (simple heuristic)
        sentences = re.split(r'[.!?។៕]', response)
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        if 20 <= avg_sentence_length <= 150:
            score = min(score + max_score * 0.2, max_score)
            feedback_parts.append("Good sentence structure")

        feedback = "; ".join(feedback_parts) if feedback_parts else "Language quality assessed"
        return score, feedback

    def _grade_technical_correctness(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str,
        max_score: float
    ) -> Tuple[float, str]:
        """Grade technical correctness of the response"""
        score = max_score * 0.5  # Base score
        feedback_parts = []

        # Check for technical terms
        tech_terms = criterion.get('required_technical_terms', [])
        if tech_terms:
            found_terms = sum(1 for term in tech_terms if term.lower() in response.lower())
            term_ratio = found_terms / len(tech_terms)
            score = max_score * term_ratio
            feedback_parts.append(f"Found {found_terms}/{len(tech_terms)} technical terms")

        # Check for proper formatting (e.g., currency, percentages)
        if re.search(r'\$[\d,]+', response) or re.search(r'[\d,]+\s*(USD|KHR|៛)', response):
            score += max_score * 0.1
            feedback_parts.append("Proper currency formatting")

        if re.search(r'\d+\.?\d*\s*%', response):
            score += max_score * 0.1
            feedback_parts.append("Contains percentage values")

        score = min(score, max_score)  # Cap at max score
        feedback = "; ".join(feedback_parts) if feedback_parts else "Technical assessment done"
        return score, feedback

    def _grade_default(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str,
        max_score: float
    ) -> Tuple[float, str]:
        """Default grading for unknown criteria"""
        # Simple length-based scoring
        if len(response) > 50:
            score = max_score * 0.6
            feedback = "Response provided with adequate content"
        else:
            score = max_score * 0.3
            feedback = "Response too brief"

        return score, feedback


class BilingualGrader(AutomatedGrader):
    """Specialized grader for bilingual (Khmer-English) responses"""

    def __init__(self):
        """Initialize bilingual grader"""
        super().__init__()
        self.grader_name = "bilingual_grader"

    def _grade_language_quality(
        self,
        criterion: Dict[str, Any],
        task: EvaluationTask,
        response: str,
        max_score: float
    ) -> Tuple[float, str]:
        """
        Enhanced language quality grading for bilingual content

        Args:
            criterion: Grading criterion
            task: Evaluation task
            response: Model response
            max_score: Maximum possible score

        Returns:
            Tuple of (score, feedback)
        """
        score = 0
        feedback_parts = []

        # Analyze language distribution
        khmer_chars = sum(1 for char in response if 0x1780 <= ord(char) <= 0x17FF)
        latin_chars = sum(1 for char in response if char.isalpha() and ord(char) < 128)
        total_chars = len(response)

        if total_chars > 0:
            khmer_ratio = khmer_chars / total_chars
            latin_ratio = latin_chars / total_chars

            # Check for appropriate bilingual balance
            if 0.3 <= khmer_ratio <= 0.8 and 0.1 <= latin_ratio <= 0.5:
                score = max_score * 0.9
                feedback_parts.append(f"Good bilingual balance (Khmer: {khmer_ratio*100:.1f}%, English: {latin_ratio*100:.1f}%)")
            elif khmer_ratio > 0:
                score = max_score * 0.6
                feedback_parts.append(f"Contains bilingual content (Khmer: {khmer_ratio*100:.1f}%, English: {latin_ratio*100:.1f}%)")
            else:
                score = max_score * 0.3
                feedback_parts.append("Missing Khmer content")

        # Check for code-switching markers
        code_switch_markers = ['/', '(', ')', '-']
        if any(marker in response for marker in code_switch_markers):
            score += max_score * 0.1
            feedback_parts.append("Proper code-switching markers used")

        score = min(score, max_score)
        feedback = "; ".join(feedback_parts) if feedback_parts else "Bilingual quality assessed"
        return score, feedback