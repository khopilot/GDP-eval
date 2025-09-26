"""
GDPval-Style Evaluation Framework for AI Model Testing
A comprehensive implementation for evaluating LLMs on real-world economically valuable tasks
"""

import json
import asyncio
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
import uuid
import logging
from abc import ABC, abstractmethod
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Data Structures ====================

class TaskCategory(Enum):
    """Categories based on GDP-contributing sectors"""
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    PROFESSIONAL_SERVICES = "professional_services"
    REAL_ESTATE = "real_estate"
    EDUCATION = "education"
    GOVERNMENT = "government"


class FileType(Enum):
    """Supported file types for multi-modal tasks"""
    TEXT = "text"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CAD = "cad"
    CODE = "code"
    JSON = "json"
    CSV = "csv"


@dataclass
class ReferenceFile:
    """Reference file for a task"""
    file_id: str
    file_name: str
    file_type: FileType
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def load_content(self) -> Any:
        """Load file content based on file type"""
        if self.file_type == FileType.TEXT:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif self.file_type == FileType.JSON:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        elif self.file_type == FileType.CSV:
            return pd.read_csv(self.file_path)
        # Add more file type handlers as needed
        else:
            return Path(self.file_path).read_bytes()


@dataclass
class EvaluationTask:
    """Individual evaluation task"""
    task_id: str
    occupation: str
    category: TaskCategory
    task_name: str
    prompt: str
    reference_files: List[ReferenceFile]
    expected_output_type: FileType
    difficulty_level: int  # 1-5 scale
    time_estimate_minutes: float
    wage_per_hour: float  # For economic value calculation
    metadata: Dict[str, Any] = field(default_factory=dict)
    expert_solution: Optional[Any] = None
    grading_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'occupation': self.occupation,
            'category': self.category.value,
            'task_name': self.task_name,
            'prompt': self.prompt,
            'reference_files': [rf.file_name for rf in self.reference_files],
            'expected_output_type': self.expected_output_type.value,
            'difficulty_level': self.difficulty_level,
            'time_estimate_minutes': self.time_estimate_minutes,
            'wage_per_hour': self.wage_per_hour,
            'metadata': self.metadata,
            'grading_criteria': self.grading_criteria
        }


@dataclass
class TaskResult:
    """Result from model evaluation on a task"""
    task_id: str
    model_name: str
    model_output: Any
    execution_time_seconds: float
    api_cost: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradingResult:
    """Grading result for a task"""
    task_id: str
    model_name: str
    grader_type: str  # 'human' or 'automated'
    score: float  # 0-1 scale
    comparison_result: str  # 'better', 'as_good', 'worse'
    feedback: str
    grader_id: Optional[str] = None
    grading_time_seconds: Optional[float] = None
    confidence: Optional[float] = None


# ==================== Core Framework Classes ====================

class TaskLoader:
    """Load and manage evaluation tasks"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.tasks: List[EvaluationTask] = []
        
    def load_tasks_from_json(self, json_path: str) -> List[EvaluationTask]:
        """Load tasks from JSON configuration"""
        with open(json_path, 'r') as f:
            tasks_data = json.load(f)
        
        tasks = []
        for task_data in tasks_data['tasks']:
            # Load reference files
            ref_files = []
            for rf_data in task_data.get('reference_files', []):
                ref_file = ReferenceFile(
                    file_id=rf_data['file_id'],
                    file_name=rf_data['file_name'],
                    file_type=FileType(rf_data['file_type']),
                    file_path=str(self.data_dir / rf_data['file_path']),
                    metadata=rf_data.get('metadata', {})
                )
                ref_files.append(ref_file)
            
            task = EvaluationTask(
                task_id=task_data['task_id'],
                occupation=task_data['occupation'],
                category=TaskCategory(task_data['category']),
                task_name=task_data['task_name'],
                prompt=task_data['prompt'],
                reference_files=ref_files,
                expected_output_type=FileType(task_data['expected_output_type']),
                difficulty_level=task_data['difficulty_level'],
                time_estimate_minutes=task_data['time_estimate_minutes'],
                wage_per_hour=task_data['wage_per_hour'],
                metadata=task_data.get('metadata', {}),
                grading_criteria=task_data.get('grading_criteria', [])
            )
            tasks.append(task)
        
        self.tasks = tasks
        logger.info(f"Loaded {len(tasks)} tasks from {json_path}")
        return tasks
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[EvaluationTask]:
        """Filter tasks by category"""
        return [t for t in self.tasks if t.category == category]
    
    def get_tasks_by_occupation(self, occupation: str) -> List[EvaluationTask]:
        """Filter tasks by occupation"""
        return [t for t in self.tasks if t.occupation == occupation]


class ModelEvaluator(ABC):
    """Abstract base class for model evaluators"""
    
    @abstractmethod
    async def evaluate_task(self, task: EvaluationTask) -> TaskResult:
        """Evaluate a single task"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier"""
        pass


class OpenAIEvaluator(ModelEvaluator):
    """Evaluator for OpenAI models"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        self.api_key = api_key
        self.model_name = model_name
        # Initialize OpenAI client here
        
    async def evaluate_task(self, task: EvaluationTask) -> TaskResult:
        """Evaluate task using OpenAI model"""
        start_time = datetime.now()
        
        try:
            # Prepare context with reference files
            context = self._prepare_context(task)
            
            # Call OpenAI API (simplified)
            # In real implementation, use actual OpenAI client
            response = await self._call_model(context, task.prompt)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task.task_id,
                model_name=self.model_name,
                model_output=response,
                execution_time_seconds=execution_time,
                api_cost=self._calculate_cost(context, response),
                success=True
            )
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {str(e)}")
            return TaskResult(
                task_id=task.task_id,
                model_name=self.model_name,
                model_output=None,
                execution_time_seconds=0,
                api_cost=0,
                success=False,
                error_message=str(e)
            )
    
    def _prepare_context(self, task: EvaluationTask) -> str:
        """Prepare context from reference files"""
        context_parts = []
        
        for ref_file in task.reference_files:
            if ref_file.file_type in [FileType.TEXT, FileType.JSON, FileType.CSV]:
                content = ref_file.load_content()
                context_parts.append(f"File: {ref_file.file_name}\n{str(content)[:5000]}")
        
        return "\n\n".join(context_parts)
    
    async def _call_model(self, context: str, prompt: str) -> str:
        """Call OpenAI model (placeholder)"""
        # Implement actual API call
        return f"Model response for prompt: {prompt[:100]}..."
    
    def _calculate_cost(self, context: str, response: str) -> float:
        """Calculate API cost based on token usage"""
        # Simplified cost calculation
        input_tokens = len(context.split()) * 1.3
        output_tokens = len(response.split()) * 1.3
        cost_per_1k_input = 0.01
        cost_per_1k_output = 0.03
        return (input_tokens * cost_per_1k_input + output_tokens * cost_per_1k_output) / 1000
    
    def get_model_name(self) -> str:
        return self.model_name


class HumanGrader:
    """Human expert grading system"""
    
    def __init__(self, grader_id: str, expertise_areas: List[str]):
        self.grader_id = grader_id
        self.expertise_areas = expertise_areas
        
    def grade_pairwise(self, 
                       task: EvaluationTask,
                       output_a: Any,
                       output_b: Any,
                       model_a_name: str,
                       model_b_name: str) -> Tuple[GradingResult, GradingResult]:
        """Perform pairwise comparison grading"""
        # In production, this would present UI for human grading
        # For now, return placeholder results
        
        print(f"Grader {self.grader_id} comparing outputs for task {task.task_name}")
        print(f"Model A ({model_a_name}): {str(output_a)[:200]}")
        print(f"Model B ({model_b_name}): {str(output_b)[:200]}")
        
        # Simulate human decision
        winner = input("Which is better? (A/B/tie): ").strip().upper()
        
        if winner == 'A':
            result_a = GradingResult(
                task_id=task.task_id,
                model_name=model_a_name,
                grader_type='human',
                score=1.0,
                comparison_result='better',
                feedback="Superior quality",
                grader_id=self.grader_id
            )
            result_b = GradingResult(
                task_id=task.task_id,
                model_name=model_b_name,
                grader_type='human',
                score=0.0,
                comparison_result='worse',
                feedback="Inferior quality",
                grader_id=self.grader_id
            )
        elif winner == 'B':
            result_a = GradingResult(
                task_id=task.task_id,
                model_name=model_a_name,
                grader_type='human',
                score=0.0,
                comparison_result='worse',
                feedback="Inferior quality",
                grader_id=self.grader_id
            )
            result_b = GradingResult(
                task_id=task.task_id,
                model_name=model_b_name,
                grader_type='human',
                score=1.0,
                comparison_result='better',
                feedback="Superior quality",
                grader_id=self.grader_id
            )
        else:
            result_a = result_b = GradingResult(
                task_id=task.task_id,
                model_name=model_a_name,
                grader_type='human',
                score=0.5,
                comparison_result='as_good',
                feedback="Comparable quality",
                grader_id=self.grader_id
            )
        
        return result_a, result_b


class AutomatedGrader:
    """Automated grading using LLM as judge"""
    
    def __init__(self, judge_model: ModelEvaluator):
        self.judge_model = judge_model
        
    async def grade_pairwise(self,
                            task: EvaluationTask,
                            output_a: Any,
                            output_b: Any) -> Tuple[GradingResult, GradingResult]:
        """Automated pairwise comparison"""
        
        grading_prompt = self._create_grading_prompt(task, output_a, output_b)
        
        # Use judge model to evaluate
        judge_task = EvaluationTask(
            task_id=f"grade_{task.task_id}",
            occupation="grader",
            category=task.category,
            task_name=f"Grade {task.task_name}",
            prompt=grading_prompt,
            reference_files=[],
            expected_output_type=FileType.JSON,
            difficulty_level=3,
            time_estimate_minutes=5,
            wage_per_hour=50,
            grading_criteria=task.grading_criteria
        )
        
        result = await self.judge_model.evaluate_task(judge_task)
        
        # Parse judge response
        try:
            judgment = json.loads(result.model_output)
            winner = judgment.get('winner', 'tie')
            confidence = judgment.get('confidence', 0.5)
            explanation = judgment.get('explanation', '')
        except:
            winner = 'tie'
            confidence = 0.5
            explanation = 'Failed to parse judgment'
        
        # Create grading results
        if winner == 'A':
            score_a, score_b = 1.0, 0.0
            comp_a, comp_b = 'better', 'worse'
        elif winner == 'B':
            score_a, score_b = 0.0, 1.0
            comp_a, comp_b = 'worse', 'better'
        else:
            score_a = score_b = 0.5
            comp_a = comp_b = 'as_good'
        
        result_a = GradingResult(
            task_id=task.task_id,
            model_name="model_a",
            grader_type='automated',
            score=score_a,
            comparison_result=comp_a,
            feedback=explanation,
            confidence=confidence
        )
        
        result_b = GradingResult(
            task_id=task.task_id,
            model_name="model_b",
            grader_type='automated',
            score=score_b,
            comparison_result=comp_b,
            feedback=explanation,
            confidence=confidence
        )
        
        return result_a, result_b
    
    def _create_grading_prompt(self, task: EvaluationTask, output_a: Any, output_b: Any) -> str:
        """Create prompt for judge model"""
        criteria_text = "\n".join([f"- {c}" for c in task.grading_criteria])
        
        return f"""
        You are an expert judge evaluating two responses for the following task:
        
        Task: {task.task_name}
        Original Prompt: {task.prompt}
        
        Grading Criteria:
        {criteria_text}
        
        Output A:
        {str(output_a)[:2000]}
        
        Output B:
        {str(output_b)[:2000]}
        
        Please evaluate which output is better based on the criteria.
        Return your judgment as JSON:
        {{
            "winner": "A" | "B" | "tie",
            "confidence": 0.0-1.0,
            "explanation": "detailed explanation"
        }}
        """


class EvaluationOrchestrator:
    """Main orchestrator for running evaluations"""
    
    def __init__(self, 
                 task_loader: TaskLoader,
                 evaluators: List[ModelEvaluator],
                 human_graders: Optional[List[HumanGrader]] = None,
                 automated_grader: Optional[AutomatedGrader] = None):
        self.task_loader = task_loader
        self.evaluators = evaluators
        self.human_graders = human_graders or []
        self.automated_grader = automated_grader
        self.results_cache: Dict[str, TaskResult] = {}
        self.grading_cache: Dict[str, GradingResult] = {}
        
    async def run_evaluation(self, 
                            task_ids: Optional[List[str]] = None,
                            categories: Optional[List[TaskCategory]] = None,
                            parallel: bool = True) -> Dict[str, Any]:
        """Run full evaluation pipeline"""
        
        # Select tasks
        if task_ids:
            tasks = [t for t in self.task_loader.tasks if t.task_id in task_ids]
        elif categories:
            tasks = []
            for cat in categories:
                tasks.extend(self.task_loader.get_tasks_by_category(cat))
        else:
            tasks = self.task_loader.tasks
        
        logger.info(f"Running evaluation on {len(tasks)} tasks with {len(self.evaluators)} models")
        
        # Phase 1: Generate outputs from all models
        all_results = {}
        for evaluator in self.evaluators:
            model_name = evaluator.get_model_name()
            logger.info(f"Evaluating with model: {model_name}")
            
            if parallel:
                results = await self._evaluate_tasks_parallel(evaluator, tasks)
            else:
                results = await self._evaluate_tasks_sequential(evaluator, tasks)
            
            all_results[model_name] = results
        
        # Phase 2: Grading
        grading_results = await self._run_grading(tasks, all_results)
        
        # Phase 3: Analysis
        analysis = self._analyze_results(all_results, grading_results)
        
        # Save results
        self._save_results(all_results, grading_results, analysis)
        
        return {
            'task_results': all_results,
            'grading_results': grading_results,
            'analysis': analysis
        }
    
    async def _evaluate_tasks_parallel(self, 
                                      evaluator: ModelEvaluator, 
                                      tasks: List[EvaluationTask]) -> List[TaskResult]:
        """Evaluate tasks in parallel"""
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def eval_with_limit(task):
            async with semaphore:
                return await evaluator.evaluate_task(task)
        
        results = await asyncio.gather(*[eval_with_limit(t) for t in tasks])
        return results
    
    async def _evaluate_tasks_sequential(self,
                                        evaluator: ModelEvaluator,
                                        tasks: List[EvaluationTask]) -> List[TaskResult]:
        """Evaluate tasks sequentially"""
        results = []
        for task in tasks:
            result = await evaluator.evaluate_task(task)
            results.append(result)
        return results
    
    async def _run_grading(self, 
                          tasks: List[EvaluationTask],
                          all_results: Dict[str, List[TaskResult]]) -> Dict[str, List[GradingResult]]:
        """Run grading phase"""
        grading_results = {}
        
        # Pairwise comparisons between models
        model_names = list(all_results.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                
                logger.info(f"Grading {model_a} vs {model_b}")
                
                for task_idx, task in enumerate(tasks):
                    result_a = all_results[model_a][task_idx]
                    result_b = all_results[model_b][task_idx]
                    
                    if result_a.success and result_b.success:
                        # Use automated grader if available
                        if self.automated_grader:
                            grade_a, grade_b = await self.automated_grader.grade_pairwise(
                                task, result_a.model_output, result_b.model_output
                            )
                            
                            if model_a not in grading_results:
                                grading_results[model_a] = []
                            if model_b not in grading_results:
                                grading_results[model_b] = []
                            
                            grading_results[model_a].append(grade_a)
                            grading_results[model_b].append(grade_b)
        
        return grading_results
    
    def _analyze_results(self,
                        task_results: Dict[str, List[TaskResult]],
                        grading_results: Dict[str, List[GradingResult]]) -> Dict[str, Any]:
        """Analyze evaluation results"""
        analysis = {
            'model_performance': {},
            'task_statistics': {},
            'economic_impact': {}
        }
        
        # Model performance analysis
        for model_name, results in task_results.items():
            success_rate = sum(1 for r in results if r.success) / len(results)
            avg_time = np.mean([r.execution_time_seconds for r in results if r.success])
            total_cost = sum(r.api_cost for r in results)
            
            # Grading scores
            if model_name in grading_results:
                grades = grading_results[model_name]
                win_rate = sum(1 for g in grades if g.comparison_result == 'better') / len(grades)
                avg_score = np.mean([g.score for g in grades])
            else:
                win_rate = avg_score = 0
            
            analysis['model_performance'][model_name] = {
                'success_rate': success_rate,
                'avg_execution_time': avg_time,
                'total_api_cost': total_cost,
                'win_rate': win_rate,
                'avg_score': avg_score
            }
        
        # Task difficulty analysis
        task_success_rates = {}
        for task in self.task_loader.tasks:
            successes = []
            for model_results in task_results.values():
                task_result = next((r for r in model_results if r.task_id == task.task_id), None)
                if task_result:
                    successes.append(task_result.success)
            
            task_success_rates[task.task_id] = {
                'difficulty': task.difficulty_level,
                'category': task.category.value,
                'success_rate': np.mean(successes) if successes else 0
            }
        
        analysis['task_statistics'] = task_success_rates
        
        # Economic impact calculation
        for task in self.task_loader.tasks:
            human_cost = (task.time_estimate_minutes / 60) * task.wage_per_hour
            
            # Find best model performance
            best_model_time = float('inf')
            best_model_cost = float('inf')
            
            for model_name, results in task_results.items():
                task_result = next((r for r in results if r.task_id == task.task_id), None)
                if task_result and task_result.success:
                    if task_result.execution_time_seconds < best_model_time:
                        best_model_time = task_result.execution_time_seconds
                        best_model_cost = task_result.api_cost
            
            if best_model_time < float('inf'):
                time_savings = task.time_estimate_minutes * 60 - best_model_time
                cost_savings = human_cost - best_model_cost
                
                analysis['economic_impact'][task.task_id] = {
                    'human_cost': human_cost,
                    'ai_cost': best_model_cost,
                    'cost_savings': cost_savings,
                    'time_savings_seconds': time_savings,
                    'roi': (cost_savings / human_cost) * 100 if human_cost > 0 else 0
                }
        
        return analysis
    
    def _save_results(self, 
                     task_results: Dict[str, List[TaskResult]],
                     grading_results: Dict[