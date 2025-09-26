"""
Advanced Grading System for GDPval Khmer Evaluation
Implements both automated and human grading with pairwise comparison
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd
from abc import ABC, abstractmethod
import hashlib
import os

logger = logging.getLogger(__name__)


class GradingMethod(Enum):
    """Grading method types"""
    PAIRWISE = "pairwise_comparison"
    ABSOLUTE = "absolute_scoring"
    REFERENCE = "reference_based"
    HYBRID = "hybrid"


class ComparisonResult(Enum):
    """Result of pairwise comparison"""
    A_BETTER = "A"
    B_BETTER = "B"
    TIE = "tie"


@dataclass
class GradingCriterion:
    """Individual grading criterion"""
    criterion_id: str
    name: str
    name_khmer: str
    description: str
    weight: float
    evaluation_type: str  # "objective", "subjective", "mixed"
    sub_criteria: Optional[List[Dict]] = None
    
    def evaluate(self, output: str, reference: Optional[str] = None) -> float:
        """Evaluate output against this criterion"""
        # This would be implemented based on evaluation_type
        pass


@dataclass
class PairwiseComparison:
    """Result of pairwise comparison between two outputs"""
    task_id: str
    model_a: str
    model_b: str
    output_a: str
    output_b: str
    winner: ComparisonResult
    confidence: float
    criterion_scores: Dict[str, Dict[str, float]]  # {criterion_id: {model_a: score, model_b: score}}
    explanation: str
    grader_id: str
    grading_time: datetime
    metadata: Dict[str, Any]


@dataclass
class AbsoluteGrade:
    """Absolute grading result for a single output"""
    task_id: str
    model: str
    output: str
    total_score: float
    criterion_scores: Dict[str, float]
    feedback: str
    grader_id: str
    grading_time: datetime
    confidence: float
    metadata: Dict[str, Any]


class BaseGrader(ABC):
    """Abstract base class for graders"""
    
    @abstractmethod
    async def grade(self, task: Dict, output: str, **kwargs) -> Any:
        """Grade a single output"""
        pass
    
    @abstractmethod
    async def compare(self, task: Dict, output_a: str, output_b: str, **kwargs) -> PairwiseComparison:
        """Compare two outputs"""
        pass


class AutomatedGrader(BaseGrader):
    """Automated grading using LLM as judge"""
    
    def __init__(self, judge_model: str = "gpt-4", temperature: float = 0.1):
        self.judge_model = judge_model
        self.temperature = temperature
        self.grading_cache = {}
        self._initialize_judge()
        
    def _initialize_judge(self):
        """Initialize the judge model"""
        if self.judge_model.startswith("gpt"):
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.judge_model.startswith("claude"):
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported judge model: {self.judge_model}")
    
    async def grade(self, task: Dict, output: str, reference: Optional[str] = None) -> AbsoluteGrade:
        """Grade a single output with absolute scoring"""
        
        # Create grading prompt
        prompt = self._create_absolute_grading_prompt(task, output, reference)
        
        # Get judgment from model
        judgment = await self._get_judgment(prompt)
        
        # Parse judgment
        scores = judgment.get("scores", {})
        total_score = self._calculate_weighted_score(scores, task.get("grading_criteria", []))
        
        return AbsoluteGrade(
            task_id=task["task_id"],
            model="unknown",
            output=output,
            total_score=total_score,
            criterion_scores=scores,
            feedback=judgment.get("feedback", ""),
            grader_id=f"auto_{self.judge_model}",
            grading_time=datetime.now(),
            confidence=judgment.get("confidence", 0.5),
            metadata=judgment.get("metadata", {})
        )
    
    async def compare(self, task: Dict, output_a: str, output_b: str, 
                     model_a: str = "model_a", model_b: str = "model_b") -> PairwiseComparison:
        """Compare two outputs using pairwise comparison"""
        
        # Check cache
        cache_key = self._get_cache_key(task["task_id"], output_a, output_b)
        if cache_key in self.grading_cache:
            logger.info(f"Using cached grading for task {task['task_id']}")
            return self.grading_cache[cache_key]
        
        # Create comparison prompt
        prompt = self._create_comparison_prompt(task, output_a, output_b)
        
        # Get judgment
        judgment = await self._get_judgment(prompt)
        
        # Parse result
        winner_str = judgment.get("winner", "tie")
        winner = ComparisonResult.A_BETTER if winner_str == "A" else \
                ComparisonResult.B_BETTER if winner_str == "B" else \
                ComparisonResult.TIE
        
        result = PairwiseComparison(
            task_id=task["task_id"],
            model_a=model_a,
            model_b=model_b,
            output_a=output_a[:500],  # Truncate for storage
            output_b=output_b[:500],
            winner=winner,
            confidence=judgment.get("confidence", 0.5),
            criterion_scores=judgment.get("criterion_scores", {}),
            explanation=judgment.get("explanation", ""),
            grader_id=f"auto_{self.judge_model}",
            grading_time=datetime.now(),
            metadata={
                "prompt_tokens": judgment.get("prompt_tokens", 0),
                "completion_tokens": judgment.get("completion_tokens", 0),
                "grading_temperature": self.temperature
            }
        )
        
        # Cache result
        self.grading_cache[cache_key] = result
        
        return result
    
    def _create_absolute_grading_prompt(self, task: Dict, output: str, reference: Optional[str]) -> str:
        """Create prompt for absolute grading"""
        
        # Extract grading criteria
        criteria_text = self._format_criteria(task.get("grading_criteria", []))
        
        # Build prompt
        prompt_parts = [
            "You are an expert evaluator for Khmer language AI models.",
            f"\nTask: {task['prompt']['instruction']}",
            f"\nTask Category: {task.get('category', 'general')}",
            f"\nExpected Output Format: {task['prompt'].get('output_format', 'text')}",
            "\n\nGrading Criteria:",
            criteria_text,
            "\n\nModel Output:",
            output[:3000],  # Limit length
        ]
        
        if reference:
            prompt_parts.extend([
                "\n\nReference Solution:",
                reference[:2000]
            ])
        
        prompt_parts.extend([
            "\n\nPlease evaluate the output based on the criteria above.",
            "Return your evaluation as JSON:",
            """{
    "scores": {
        "criterion_id": score (0.0-1.0),
        ...
    },
    "feedback": "detailed feedback in English",
    "feedback_khmer": "មតិយោបល់លម្អិតជាភាសាខ្មែរ",
    "confidence": 0.0-1.0,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "metadata": {
        "language_quality": "assessment of Khmer language quality",
        "technical_accuracy": "assessment of technical accuracy"
    }
}"""
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_comparison_prompt(self, task: Dict, output_a: str, output_b: str) -> str:
        """Create prompt for pairwise comparison"""
        
        criteria_text = self._format_criteria(task.get("grading_criteria", []))
        
        # Include bilingual instructions
        prompt = f"""You are an expert evaluator comparing two AI model outputs for a Khmer language task.

**Task Instructions:**
{task['prompt']['instruction']}

**Task Instructions (English):**
{task['prompt'].get('instruction_english', 'N/A')}

**Category:** {task.get('category', 'general')}
**Occupation:** {task.get('occupation', 'general')}

**Grading Criteria:**
{criteria_text}

**Output A:**
{output_a[:2500]}

**Output B:**
{output_b[:2500]}

Please compare these outputs comprehensively, considering:
1. Task completion and accuracy
2. Khmer language quality (grammar, fluency, appropriateness)
3. Technical terminology usage (if applicable)
4. Code-switching appropriateness (Khmer-English balance)
5. Professional tone and formatting
6. Adherence to output format requirements

Return your judgment as JSON:
{{
    "winner": "A" | "B" | "tie",
    "confidence": 0.0-1.0,
    "explanation": "detailed explanation in English",
    "explanation_khmer": "ការពន្យល់លម្អិតជាភាសាខ្មែរ",
    "criterion_scores": {{
        "criterion_id": {{
            "A": score,
            "B": score,
            "explanation": "why this score"
        }},
        ...
    }},
    "a_strengths": ["list of strengths for A"],
    "a_weaknesses": ["list of weaknesses for A"],
    "b_strengths": ["list of strengths for B"],
    "b_weaknesses": ["list of weaknesses for B"],
    "language_assessment": {{
        "a_fluency": 0.0-1.0,
        "b_fluency": 0.0-1.0,
        "a_terminology": 0.0-1.0,
        "b_terminology": 0.0-1.0
    }}
}}"""
        
        return prompt
    
    def _format_criteria(self, criteria: List[Dict]) -> str:
        """Format grading criteria for prompt"""
        if not criteria:
            return "General quality assessment"
        
        formatted = []
        for i, criterion in enumerate(criteria, 1):
            formatted.append(
                f"{i}. **{criterion.get('criterion_name', criterion['criterion_id'])}** "
                f"(Weight: {criterion['weight']:.0%})\n"
                f"   - {criterion['description']}"
            )
            
            if criterion.get('evaluation_points'):
                formatted.append("   Evaluation points:")
                for point in criterion['evaluation_points']:
                    formatted.append(f"     • {point}")
        
        return "\n".join(formatted)
    
    async def _get_judgment(self, prompt: str) -> Dict:
        """Get judgment from the judge model"""
        try:
            if self.judge_model.startswith("gpt"):
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator fluent in Khmer and English."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                judgment = json.loads(content)
                
                # Add token usage
                judgment["prompt_tokens"] = response.usage.prompt_tokens
                judgment["completion_tokens"] = response.usage.completion_tokens
                
            elif self.judge_model.startswith("claude"):
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=2000
                )
                
                content = response.content[0].text
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    judgment = json.loads(json_match.group())
                else:
                    judgment = {"error": "Failed to parse response"}
            
            else:
                judgment = {"error": f"Unsupported model: {self.judge_model}"}
            
            return judgment
            
        except Exception as e:
            logger.error(f"Error getting judgment: {e}")
            return {
                "winner": "tie",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "criterion_scores": {}
            }
    
    def _calculate_weighted_score(self, scores: Dict[str, float], criteria: List[Dict]) -> float:
        """Calculate weighted total score"""
        if not criteria:
            return np.mean(list(scores.values())) if scores else 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for criterion in criteria:
            crit_id = criterion['criterion_id']
            weight = criterion['weight']
            
            if crit_id in scores:
                weighted_sum += scores[crit_id] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_cache_key(self, task_id: str, output_a: str, output_b: str) -> str:
        """Generate cache key for grading result"""
        content = f"{task_id}:{output_a[:500]}:{output_b[:500]}"
        return hashlib.md5(content.encode()).hexdigest()


class HumanGrader(BaseGrader):
    """Human expert grading interface"""
    
    def __init__(self, grader_id: str, expertise: List[str], interface_type: str = "cli"):
        self.grader_id = grader_id
        self.expertise = expertise
        self.interface_type = interface_type
        self.grading_history = []
        
    async def grade(self, task: Dict, output: str, reference: Optional[str] = None) -> AbsoluteGrade:
        """Get absolute grade from human expert"""
        
        if self.interface_type == "cli":
            return await self._grade_cli(task, output, reference)
        elif self.interface_type == "web":
            return await self._grade_web(task, output, reference)
        else:
            raise ValueError(f"Unknown interface type: {self.interface_type}")
    
    async def compare(self, task: Dict, output_a: str, output_b: str,
                      model_a: str = "model_a", model_b: str = "model_b") -> PairwiseComparison:
        """Get pairwise comparison from human expert"""
        
        if self.interface_type == "cli":
            return await self._compare_cli(task, output_a, output_b, model_a, model_b)
        elif self.interface_type == "web":
            return await self._compare_web(task, output_a, output_b, model_a, model_b)
        else:
            raise ValueError(f"Unknown interface type: {self.interface_type}")
    
    async def _grade_cli(self, task: Dict, output: str, reference: Optional[str]) -> AbsoluteGrade:
        """CLI interface for absolute grading"""
        
        print("\n" + "=" * 80)
        print(f"HUMAN GRADING - Task: {task['task_id']}")
        print("=" * 80)
        
        print(f"\nTask: {task['prompt']['instruction'][:500]}")
        print(f"\nCategory: {task.get('category', 'general')}")
        print(f"\nOutput to Grade:\n{output[:1000]}")
        
        if reference:
            print(f"\nReference Solution:\n{reference[:500]}")
        
        print("\nGrading Criteria:")
        for i, criterion in enumerate(task.get('grading_criteria', []), 1):
            print(f"{i}. {criterion['criterion_name']} (Weight: {criterion['weight']:.0%})")
            print(f"   {criterion['description']}")
        
        # Collect scores
        scores = {}
        for criterion in task.get('grading_criteria', []):
            while True:
                try:
                    score = float(input(f"\nScore for {criterion['criterion_name']} (0-1): "))
                    if 0 <= score <= 1:
                        scores[criterion['criterion_id']] = score
                        break
                    print("Score must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")
        
        feedback = input("\nFeedback (optional): ").strip()
        confidence = float(input("Confidence in grading (0-1): "))
        
        total_score = self._calculate_weighted_score(scores, task.get('grading_criteria', []))
        
        grade = AbsoluteGrade(
            task_id=task['task_id'],
            model="evaluated_model",
            output=output[:500],
            total_score=total_score,
            criterion_scores=scores,
            feedback=feedback,
            grader_id=self.grader_id,
            grading_time=datetime.now(),
            confidence=confidence,
            metadata={"interface": "cli"}
        )
        
        self.grading_history.append(grade)
        return grade
    
    async def _compare_cli(self, task: Dict, output_a: str, output_b: str,
                           model_a: str, model_b: str) -> PairwiseComparison:
        """CLI interface for pairwise comparison"""
        
        print("\n" + "=" * 80)
        print(f"PAIRWISE COMPARISON - Task: {task['task_id']}")
        print("=" * 80)
        
        print(f"\nTask: {task['prompt']['instruction'][:500]}")
        
        print("\n" + "-" * 40)
        print("OUTPUT A:")
        print("-" * 40)
        print(output_a[:1000])
        
        print("\n" + "-" * 40)
        print("OUTPUT B:")
        print("-" * 40)
        print(output_b[:1000])
        
        print("\nGrading Criteria:")
        for i, criterion in enumerate(task.get('grading_criteria', []), 1):
            print(f"{i}. {criterion['criterion_name']} - {criterion['description']}")
        
        # Get comparison
        while True:
            winner_input = input("\nWhich output is better? (A/B/tie): ").strip().upper()
            if winner_input in ['A', 'B', 'TIE']:
                break
            print("Please enter A, B, or tie")
        
        winner = ComparisonResult.A_BETTER if winner_input == 'A' else \
                ComparisonResult.B_BETTER if winner_input == 'B' else \
                ComparisonResult.TIE
        
        explanation = input("\nExplain your choice: ").strip()
        confidence = float(input("Confidence (0-1): "))
        
        # Optional: detailed criterion scores
        criterion_scores = {}
        if input("\nProvide detailed scores? (y/n): ").lower() == 'y':
            for criterion in task.get('grading_criteria', []):
                crit_id = criterion['criterion_id']
                score_a = float(input(f"Score for A on {criterion['criterion_name']} (0-1): "))
                score_b = float(input(f"Score for B on {criterion['criterion_name']} (0-1): "))
                criterion_scores[crit_id] = {"A": score_a, "B": score_b}
        
        comparison = PairwiseComparison(
            task_id=task['task_id'],
            model_a=model_a,
            model_b=model_b,
            output_a=output_a[:500],
            output_b=output_b[:500],
            winner=winner,
            confidence=confidence,
            criterion_scores=criterion_scores,
            explanation=explanation,
            grader_id=self.grader_id,
            grading_time=datetime.now(),
            metadata={"interface": "cli"}
        )
        
        self.grading_history.append(comparison)
        return comparison
    
    async def _grade_web(self, task: Dict, output: str, reference: Optional[str]) -> AbsoluteGrade:
        """Web interface for grading (placeholder)"""
        # This would integrate with a web UI
        raise NotImplementedError("Web interface not yet implemented")
    
    async def _compare_web(self, task: Dict, output_a: str, output_b: str,
                           model_a: str, model_b: str) -> PairwiseComparison:
        """Web interface for comparison (placeholder)"""
        raise NotImplementedError("Web interface not yet implemented")
    
    def _calculate_weighted_score(self, scores: Dict[str, float], criteria: List[Dict]) -> float:
        """Calculate weighted total score"""
        if not criteria:
            return np.mean(list(scores.values())) if scores else 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for criterion in criteria:
            crit_id = criterion['criterion_id']
            weight = criterion['weight']
            
            if crit_id in scores:
                weighted_sum += scores[crit_id] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class EnsembleGrader:
    """Ensemble grading combining multiple graders"""
    
    def __init__(self, graders: List[BaseGrader], aggregation_method: str = "weighted_average"):
        self.graders = graders
        self.aggregation_method = aggregation_method
        
    async def grade_ensemble(self, task: Dict, output: str) -> Dict:
        """Get ensemble grade from multiple graders"""
        
        grades = []
        
        # Collect grades from all graders
        for grader in self.graders:
            try:
                grade = await grader.grade(task, output)
                grades.append(grade)
            except Exception as e:
                logger.error(f"Error with grader {grader}: {e}")
        
        if not grades:
            raise ValueError("No successful grades obtained")
        
        # Aggregate grades
        if self.aggregation_method == "weighted_average":
            return self._aggregate_weighted(grades)
        elif self.aggregation_method == "median":
            return self._aggregate_median(grades)
        elif self.aggregation_method == "majority_vote":
            return self._aggregate_majority(grades)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _aggregate_weighted(self, grades: List[AbsoluteGrade]) -> Dict:
        """Weighted average aggregation"""
        
        # Weight by grader confidence
        weights = [g.confidence for g in grades]
        total_weight = sum(weights)
        
        if total_weight == 0:
            weights = [1.0] * len(grades)
            total_weight = len(grades)
        
        # Aggregate scores
        aggregated_scores = {}
        all_criteria = set()
        
        for grade in grades:
            all_criteria.update(grade.criterion_scores.keys())
        
        for criterion in all_criteria:
            weighted_sum = 0
            weight_sum = 0
            
            for grade, weight in zip(grades, weights):
                if criterion in grade.criterion_scores:
                    weighted_sum += grade.criterion_scores[criterion] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                aggregated_scores[criterion] = weighted_sum / weight_sum
        
        # Calculate total score
        total_score = np.average(
            [g.total_score for g in grades],
            weights=weights
        )
        
        return {
            "aggregation_method": "weighted_average",
            "total_score": total_score,
            "criterion_scores": aggregated_scores,
            "num_graders": len(grades),
            "confidence": np.mean([g.confidence for g in grades]),
            "individual_grades": [asdict(g) for g in grades]
        }
    
    def _aggregate_median(self, grades: List[AbsoluteGrade]) -> Dict:
        """Median aggregation"""
        
        # Calculate median for each criterion
        aggregated_scores = {}
        all_criteria = set()
        
        for grade in grades:
            all_criteria.update(grade.criterion_scores.keys())
        
        for criterion in all_criteria:
            scores = [g.criterion_scores.get(criterion, 0) for g in grades]
            aggregated_scores[criterion] = np.median(scores)
        
        # Median total score
        total_score = np.median([g.total_score for g in grades])
        
        return {
            "aggregation_method": "median",
            "total_score": total_score,
            "criterion_scores": aggregated_scores,
            "num_graders": len(grades),
            "confidence": np.median([g.confidence for g in grades]),
            "individual_grades": [asdict(g) for g in grades]
        }
    
    def _aggregate_majority(self, grades: List[AbsoluteGrade]) -> Dict:
        """Majority vote aggregation for pairwise comparisons"""
        # This would be used for PairwiseComparison results
        pass


class CalibrationAnalyzer:
    """Analyze and calibrate grading consistency"""
    
    def __init__(self):
        self.calibration_data = []
        
    def add_calibration_sample(self, human_grade: Any, auto_grade: Any):
        """Add calibration sample"""
        self.calibration_data.append({
            "human": human_grade,
            "auto": auto_grade,
            "timestamp": datetime.now()
        })
    
    def calculate_agreement(self) -> Dict[str, float]:
        """Calculate inter-rater agreement metrics"""
        
        if len(self.calibration_data) < 2:
            return {"error": "Insufficient calibration data"}
        
        # Extract scores
        human_scores = []
        auto_scores = []
        
        for sample in self.calibration_data:
            if hasattr(sample["human"], "total_score"):
                human_scores.append(sample["human"].total_score)
                auto_scores.append(sample["auto"].total_score)
        
        if not human_scores:
            return {"error": "No valid scores found"}
        
        # Calculate agreement metrics
        correlation = np.corrcoef(human_scores, auto_scores)[0, 1]
        
        # Cohen's Kappa for binned scores
        human_bins = np.digitize(human_scores, bins=[0, 0.33, 0.67, 1.0])
        auto_bins = np.digitize(auto_scores, bins=[0, 0.33, 0.67, 1.0])
        
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(human_bins, auto_bins)
        
        # Mean absolute error
        mae = np.mean(np.abs(np.array(human_scores) - np.array(auto_scores)))
        
        return {
            "pearson_correlation": correlation,
            "cohens_kappa": kappa,
            "mean_absolute_error": mae,
            "num_samples": len(human_scores),
            "human_mean": np.mean(human_scores),
            "human_std": np.std(human_scores),
            "auto_mean": np.mean(auto_scores),
            "auto_std": np.std(auto_scores)
        }
    
    def get_calibration_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibration curve for visualization"""
        
        if len(self.calibration_data) < 10:
            return np.array([]), np.array([])
        
        human_scores = []
        auto_scores = []
        
        for sample in self.calibration_data:
            if hasattr(sample["human"], "total_score"):
                human_scores.append(sample["human"].total_score)
                auto_scores.append(sample["auto"].total_score)
        
        # Bin scores and calculate means
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(human_scores, bins)
        
        calibration_x = []
        calibration_y = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                calibration_x.append(np.mean(np.array(human_scores)[mask]))
                calibration_y.append(np.mean(np.array(auto_scores)[mask]))
        
        return np.array(calibration_x), np.array(calibration_y)


# Main grading orchestrator
class GradingOrchestrator:
    """Orchestrates the entire grading process"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.automated_grader = None
        self.human_graders = []
        self.calibration_analyzer = CalibrationAnalyzer()
        
        self._initialize_graders()
        
    def _initialize_graders(self):
        """Initialize configured graders"""
        
        if self.config.get("use_automated_grading"):
            self.automated_grader = AutomatedGrader(
                judge_model=self.config.get("judge_model", "gpt-4"),
                temperature=self.config.get("temperature", 0.1)
            )
        
        if self.config.get("use_human_grading"):
            # Initialize human graders
            for grader_config in self.config.get("human_graders", []):
                grader = HumanGrader(
                    grader_id=grader_config["id"],
                    expertise=grader_config["expertise"],
                    interface_type=grader_config.get("interface", "cli")
                )
                self.human_graders.append(grader)
    
    async def grade_batch(self, tasks: List[Dict], results: List[Dict]) -> Dict:
        """Grade a batch of task results"""
        
        all_grades = {
            "absolute_grades": [],
            "pairwise_comparisons": [],
            "metadata": {
                "grading_started": datetime.now().isoformat(),
                "config": self.config
            }
        }
        
        # Group results by task
        results_by_task = {}
        for result in results:
            task_id = result["task_id"]
            if task_id not in results_by_task:
                results_by_task[task_id] = []
            results_by_task[task_id].append(result)
        
        # Grade each task
        for task in tasks:
            task_id = task["task_id"]
            
            if task_id not in results_by_task:
                logger.warning(f"No results found for task {task_id}")
                continue
            
            task_results = results_by_task[task_id]
            
            # Absolute grading
            if self.config.get("use_absolute_grading"):
                for result in task_results:
                    if result.get("success") and result.get("response"):
                        grade = await self._grade_absolute(task, result)
                        if grade:
                            all_grades["absolute_grades"].append(asdict(grade))
            
            # Pairwise comparisons
            if self.config.get("use_pairwise_comparison") and len(task_results) > 1:
                comparisons = await self._grade_pairwise(task, task_results)
                for comp in comparisons:
                    all_grades["pairwise_comparisons"].append(asdict(comp))
        
        all_grades["metadata"]["grading_completed"] = datetime.now().isoformat()
        
        # Calculate aggregate statistics
        all_grades["statistics"] = self._calculate_statistics(all_grades)
        
        return all_grades
    
    async def _grade_absolute(self, task: Dict, result: Dict) -> Optional[AbsoluteGrade]:
        """Grade single result with absolute scoring"""
        
        grades = []
        
        # Automated grading
        if self.automated_grader:
            try:
                auto_grade = await self.automated_grader.grade(
                    task,
                    result["response"],
                    task.get("expert_solution", {}).get("content")
                )
                auto_grade.model = result.get("model_type", "unknown")
                grades.append(auto_grade)
            except Exception as e:
                logger.error(f"Automated grading failed: {e}")
        
        # Human grading (if configured)
        if self.human_graders and self.config.get("sample_for_human_grading", 0) > np.random.random():
            for grader in self.human_graders[:1]:  # Use first available human grader
                try:
                    human_grade = await grader.grade(
                        task,
                        result["response"],
                        task.get("expert_solution", {}).get("content")
                    )
                    human_grade.model = result.get("model_type", "unknown")
                    grades.append(human_grade)
                    
                    # Add to calibration data
                    if self.automated_grader and len(grades) > 1:
                        self.calibration_analyzer.add_calibration_sample(
                            human_grade,
                            grades[0]  # Assuming first is auto grade
                        )
                except Exception as e:
                    logger.error(f"Human grading failed: {e}")
        
        # Return best grade or ensemble
        if grades:
            if len(grades) == 1:
                return grades[0]
            else:
                # Could implement ensemble here
                return grades[0]  # For now, return automated grade
        
        return None
    
    async def _grade_pairwise(self, task: Dict, results: List[Dict]) -> List[PairwiseComparison]:
        """Grade results using pairwise comparison"""
        
        comparisons = []
        
        # Compare all pairs
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not (results[i].get("success") and results[j].get("success")):
                    continue
                
                # Automated comparison
                if self.automated_grader:
                    try:
                        comp = await self.automated_grader.compare(
                            task,
                            results[i]["response"],
                            results[j]["response"],
                            results[i].get("model_type", f"model_{i}"),
                            results[j].get("model_type", f"model_{j}")
                        )
                        comparisons.append(comp)
                    except Exception as e:
                        logger.error(f"Pairwise comparison failed: {e}")
                
                # Human comparison (sampling)
                if self.human_graders and self.config.get("sample_for_human_comparison", 0) > np.random.random():
                    for grader in self.human_graders[:1]:
                        try:
                            human_comp = await grader.compare(
                                task,
                                results[i]["response"],
                                results[j]["response"],
                                results[i].get("model_type", f"model_{i}"),
                                results[j].get("model_type", f"model_{j}")
                            )
                            comparisons.append(human_comp)
                        except Exception as e:
                            logger.error(f"Human comparison failed: {e}")
        
        return comparisons
    
    def _calculate_statistics(self, grades: Dict) -> Dict:
        """Calculate grading statistics"""
        
        stats = {
            "total_absolute_grades": len(grades["absolute_grades"]),
            "total_pairwise_comparisons": len(grades["pairwise_comparisons"])
        }
        
        # Absolute grade statistics
        if grades["absolute_grades"]:
            scores = [g["total_score"] for g in grades["absolute_grades"]]
            stats["absolute_grades"] = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "median_score": np.median(scores)
            }
        
        # Pairwise comparison statistics
        if grades["pairwise_comparisons"]:
            # Calculate win rates
            model_wins = {}
            model_comparisons = {}
            
            for comp in grades["pairwise_comparisons"]:
                model_a = comp["model_a"]
                model_b = comp["model_b"]
                winner = comp["winner"]
                
                # Initialize counters
                for model in [model_a, model_b]:
                    if model not in model_wins:
                        model_wins[model] = 0
                        model_comparisons[model] = 0
                
                # Count wins
                if winner == "A":
                    model_wins[model_a] += 1
                elif winner == "B":
                    model_wins[model_b] += 1
                else:  # tie
                    model_wins[model_a] += 0.5
                    model_wins[model_b] += 0.5
                
                model_comparisons[model_a] += 1
                model_comparisons[model_b] += 1
            
            # Calculate win rates
            win_rates = {}
            for model in model_wins:
                if model_comparisons[model] > 0:
                    win_rates[model] = model_wins[model] / model_comparisons[model]
            
            stats["win_rates"] = win_rates
            
            # Confidence statistics
            confidences = [comp["confidence"] for comp in grades["pairwise_comparisons"]]
            stats["comparison_confidence"] = {
                "mean": np.mean(confidences),
                "std": np.std(confidences)
            }
        
        # Calibration statistics
        if self.calibration_analyzer.calibration_data:
            stats["calibration"] = self.calibration_analyzer.calculate_agreement()
        
        return stats


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_grading():
        """Test grading system"""
        
        # Sample task
        task = {
            "task_id": "TEST-001",
            "category": "technology",
            "prompt": {
                "instruction": "Write a Python function to count words in Khmer text",
                "instruction_english": "Write a Python function to count words in Khmer text",
                "output_format": "code"
            },
            "grading_criteria": [
                {
                    "criterion_id": "correctness",
                    "criterion_name": "Correctness",
                    "weight": 0.5,
                    "description": "Function correctly counts Khmer words"
                },
                {
                    "criterion_id": "efficiency",
                    "criterion_name": "Efficiency",
                    "weight": 0.3,
                    "description": "Code is efficient"
                },
                {
                    "criterion_id": "style",
                    "criterion_name": "Code Style",
                    "weight": 0.2,
                    "description": "Clean, readable code"
                }
            ]
        }
        
        # Sample outputs
        output_a = """
def count_khmer_words(text):
    # Simple word counting for Khmer text
    import re
    words = re.findall(r'[\u1780-\u17FF]+', text)
    return len(words)
"""
        
        output_b = """
def count_words(txt):
    cnt = 0
    for c in txt:
        if ord(c) >= 0x1780 and ord(c) <= 0x17FF:
            cnt = cnt + 1
    return cnt
"""
        
        # Test automated grader
        print("Testing Automated Grader...")
        auto_grader = AutomatedGrader(judge_model="gpt-4")
        
        # Test absolute grading
        grade = await auto_grader.grade(task, output_a)
        print(f"Absolute Grade: {grade.total_score:.2f}")
        print(f"Criterion Scores: {grade.criterion_scores}")
        
        # Test pairwise comparison
        comparison = await auto_grader.compare(task, output_a, output_b)
        print(f"\nPairwise Comparison:")
        print(f"Winner: {comparison.winner.value}")
        print(f"Confidence: {comparison.confidence:.2f}")
        print(f"Explanation: {comparison.explanation}")
        
        # Test orchestrator
        print("\nTesting Grading Orchestrator...")
        config = {
            "use_automated_grading": True,
            "use_absolute_grading": True,
            "use_pairwise_comparison": True,
            "judge_model": "gpt-4",
            "temperature": 0.1
        }
        
        orchestrator = GradingOrchestrator(config)
        
        results = [
            {
                "task_id": "TEST-001",
                "model_type": "model_a",
                "response": output_a,
                "success": True
            },
            {
                "task_id": "TEST-001",
                "model_type": "model_b",
                "response": output_b,
                "success": True
            }
        ]
        
        grades = await orchestrator.grade_batch([task], results)
        print(f"\nTotal Grades: {grades['statistics']}")
    
    # Run test
    # asyncio.run(test_grading())
    print("Grading system module loaded successfully")