#!/usr/bin/env python3
"""
Main GDPval Evaluation Script for Khmer Language Models
Runs comprehensive evaluation pipeline with economic impact analysis
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import torch
import yaml
from tqdm import tqdm
import aiofiles
import hashlib
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules (these would be in src/)
try:
    from src.core.task_loader import KhmerTaskLoader
    from src.core.evaluator import KhmerModelEvaluator, BaselineEvaluator
    from src.core.grader import BilingualGrader, AutomatedGrader
    from src.analysis.performance import PerformanceAnalyzer
    from src.analysis.economic_impact import CambodianEconomicAnalyzer
    from src.models.khmer_model import KhmerLLM
    from src.utils.khmer_utils import KhmerTextProcessor
    from src.utils.file_handlers import MultiModalFileHandler
except ImportError as e:
    logger.warning(f"Import error (expected in standalone mode): {e}")


# ==================== Configuration ====================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation run"""
    model_path: str
    task_categories: List[str]
    data_dir: str
    output_dir: str
    use_human_grading: bool = False
    use_automated_grading: bool = True
    baseline_models: List[str] = None
    parallel_execution: bool = True
    max_concurrent_tasks: int = 10
    save_intermediate: bool = True
    language_check: bool = True
    economic_analysis: bool = True
    verbose: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


# ==================== Task Management ====================

class GDPvalTaskManager:
    """Manages task loading and preprocessing for evaluation"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.tasks = []
        self.task_index = {}
        self.khmer_processor = KhmerTextProcessor()
        
    def load_tasks(self, categories: Optional[List[str]] = None) -> List[Dict]:
        """Load evaluation tasks from JSON files"""
        logger.info(f"Loading tasks from {self.data_dir}")
        
        task_files = list(self.data_dir.glob("**/*.json"))
        all_tasks = []
        
        for task_file in tqdm(task_files, desc="Loading tasks"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'tasks' in data:
                    tasks = data['tasks']
                else:
                    tasks = [data]
                
                for task in tasks:
                    # Filter by category if specified
                    if categories and task.get('category') not in categories:
                        continue
                    
                    # Validate Khmer text if present
                    if self._contains_khmer(task.get('prompt', {}).get('instruction', '')):
                        if not self.khmer_processor.validate_text(task['prompt']['instruction']):
                            logger.warning(f"Invalid Khmer text in task {task.get('task_id', 'unknown')}")
                            continue
                    
                    # Load reference files
                    task['reference_files_loaded'] = self._load_reference_files(
                        task.get('reference_files', [])
                    )
                    
                    all_tasks.append(task)
                    self.task_index[task['task_id']] = task
                    
            except Exception as e:
                logger.error(f"Error loading {task_file}: {e}")
        
        self.tasks = all_tasks
        logger.info(f"Loaded {len(all_tasks)} tasks")
        return all_tasks
    
    def _contains_khmer(self, text: str) -> bool:
        """Check if text contains Khmer characters"""
        khmer_range = range(0x1780, 0x17FF)
        return any(ord(char) in khmer_range for char in text)
    
    def _load_reference_files(self, reference_files: List[Dict]) -> List[Dict]:
        """Load content of reference files"""
        loaded_files = []
        
        for ref_file in reference_files:
            file_path = self.data_dir / 'reference_files' / ref_file['file_name']
            
            if file_path.exists():
                ref_file['content'] = self._read_file_content(file_path, ref_file['file_type'])
                ref_file['loaded'] = True
            else:
                logger.warning(f"Reference file not found: {file_path}")
                ref_file['loaded'] = False
            
            loaded_files.append(ref_file)
        
        return loaded_files
    
    def _read_file_content(self, file_path: Path, file_type: str) -> Any:
        """Read file content based on type"""
        try:
            if file_type in ['text', 'txt']:
                return file_path.read_text(encoding='utf-8')
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_type == 'csv':
                return pd.read_csv(file_path).to_dict('records')
            elif file_type == 'spreadsheet':
                return pd.read_excel(file_path, sheet_name=None)
            else:
                return f"[Binary file: {file_path.name}]"
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None


# ==================== Model Evaluation ====================

class ModelEvaluator:
    """Handles model evaluation on tasks"""
    
    def __init__(self, model_path: str, model_type: str = "khmer"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the model for evaluation"""
        logger.info(f"Loading model from {self.model_path}")
        
        if self.model_type == "khmer":
            # Load custom Khmer model
            self.model = self._load_khmer_model()
        elif self.model_type == "openai":
            self.model = self._load_openai_model()
        elif self.model_type == "anthropic":
            self.model = self._load_anthropic_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_khmer_model(self):
        """Load fine-tuned Khmer model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        return {"model": model, "tokenizer": tokenizer}
    
    def _load_openai_model(self):
        """Initialize OpenAI API client"""
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return {"client": openai, "model_name": self.model_path}
    
    def _load_anthropic_model(self):
        """Initialize Anthropic API client"""
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return {"client": client, "model_name": self.model_path}
    
    async def evaluate_task(self, task: Dict) -> Dict:
        """Evaluate a single task"""
        start_time = datetime.now()
        
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(task)
            
            # Generate response
            if self.model_type == "khmer":
                response = await self._generate_khmer(prompt, task)
            elif self.model_type == "openai":
                response = await self._generate_openai(prompt, task)
            elif self.model_type == "anthropic":
                response = await self._generate_anthropic(prompt, task)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "task_id": task["task_id"],
                "model_type": self.model_type,
                "response": response,
                "execution_time": execution_time,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating task {task['task_id']}: {e}")
            return {
                "task_id": task["task_id"],
                "model_type": self.model_type,
                "response": None,
                "execution_time": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_prompt(self, task: Dict) -> str:
        """Prepare prompt with reference files and instructions"""
        prompt_parts = []
        
        # Add context
        if task.get('prompt', {}).get('context'):
            prompt_parts.append(f"Context: {task['prompt']['context']}")
        
        # Add reference files
        for ref_file in task.get('reference_files_loaded', []):
            if ref_file.get('loaded') and ref_file.get('content'):
                prompt_parts.append(f"\n--- {ref_file['file_name']} ---")
                content = ref_file['content']
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, ensure_ascii=False, indent=2)[:5000]
                else:
                    content = str(content)[:5000]
                prompt_parts.append(content)
        
        # Add main instruction
        prompt_parts.append(f"\nTask: {task['prompt']['instruction']}")
        
        # Add output format guidance
        if task['prompt'].get('output_format'):
            prompt_parts.append(f"\nOutput Format: {task['prompt']['output_format']}")
        
        return "\n\n".join(prompt_parts)
    
    async def _generate_khmer(self, prompt: str, task: Dict) -> str:
        """Generate response using Khmer model"""
        model = self.model["model"]
        tokenizer = self.model["tokenizer"]
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return response
    
    async def _generate_openai(self, prompt: str, task: Dict) -> str:
        """Generate response using OpenAI API"""
        client = self.model["client"]
        model_name = self.model["model_name"]
        
        response = await asyncio.to_thread(
            client.ChatCompletion.create,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant fluent in Khmer and English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
    
    async def _generate_anthropic(self, prompt: str, task: Dict) -> str:
        """Generate response using Anthropic API"""
        client = self.model["client"]
        model_name = self.model["model_name"]
        
        response = await asyncio.to_thread(
            client.messages.create,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )
        
        return response.content[0].text


# ==================== Grading System ====================

class GradingSystem:
    """Handles grading of model outputs"""
    
    def __init__(self, grading_config: Dict):
        self.config = grading_config
        self.grader_model = None
        if grading_config.get('use_automated'):
            self._initialize_grader()
    
    def _initialize_grader(self):
        """Initialize automated grader model"""
        logger.info("Initializing automated grader")
        # Initialize grading model (e.g., GPT-4)
        pass
    
    async def grade_outputs(self, task: Dict, outputs: List[Dict]) -> List[Dict]:
        """Grade model outputs for a task"""
        grading_results = []
        
        if len(outputs) < 2:
            # Single output, grade against expert solution if available
            if task.get('expert_solution'):
                result = await self._grade_against_expert(task, outputs[0])
                grading_results.append(result)
        else:
            # Pairwise comparison
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    result = await self._pairwise_comparison(task, outputs[i], outputs[j])
                    grading_results.append(result)
        
        return grading_results
    
    async def _grade_against_expert(self, task: Dict, output: Dict) -> Dict:
        """Grade output against expert solution"""
        criteria = task.get('grading_criteria', [])
        scores = {}
        
        for criterion in criteria:
            score = await self._evaluate_criterion(
                task,
                output['response'],
                task['expert_solution'],
                criterion
            )
            scores[criterion['criterion_id']] = score
        
        weighted_score = sum(
            scores[c['criterion_id']] * c['weight']
            for c in criteria
        )
        
        return {
            "task_id": task["task_id"],
            "model": output["model_type"],
            "scores": scores,
            "weighted_score": weighted_score,
            "grading_method": "expert_comparison"
        }
    
    async def _pairwise_comparison(self, task: Dict, output_a: Dict, output_b: Dict) -> Dict:
        """Perform pairwise comparison between two outputs"""
        prompt = self._create_comparison_prompt(task, output_a['response'], output_b['response'])
        
        # Use grader model to compare
        # This would call GPT-4 or another model for judgment
        judgment = await self._get_judgment(prompt)
        
        return {
            "task_id": task["task_id"],
            "model_a": output_a["model_type"],
            "model_b": output_b["model_type"],
            "winner": judgment["winner"],
            "confidence": judgment.get("confidence", 0.5),
            "explanation": judgment.get("explanation", ""),
            "grading_method": "pairwise_comparison"
        }
    
    def _create_comparison_prompt(self, task: Dict, output_a: str, output_b: str) -> str:
        """Create prompt for pairwise comparison"""
        criteria_text = "\n".join([
            f"- {c['criterion_name']}: {c['description']} (weight: {c['weight']})"
            for c in task.get('grading_criteria', [])
        ])
        
        return f"""Compare these two responses for the following task:

Task: {task['prompt']['instruction']}

Grading Criteria:
{criteria_text}

Response A:
{output_a[:3000]}

Response B:
{output_b[:3000]}

Which response is better overall? Consider all criteria and their weights.
Return your judgment as JSON:
{{
    "winner": "A" | "B" | "tie",
    "confidence": 0.0-1.0,
    "explanation": "detailed explanation",
    "criteria_scores": {{
        "A": {{"criterion_id": score, ...}},
        "B": {{"criterion_id": score, ...}}
    }}
}}
"""
    
    async def _get_judgment(self, prompt: str) -> Dict:
        """Get judgment from grader model"""
        # Placeholder - would call actual grading model
        return {
            "winner": "A",
            "confidence": 0.75,
            "explanation": "Response A provides more comprehensive analysis"
        }
    
    async def _evaluate_criterion(self, task: Dict, output: str, expert: Dict, criterion: Dict) -> float:
        """Evaluate output on a specific criterion"""
        # Placeholder - would implement actual evaluation logic
        return np.random.uniform(0.6, 1.0)


# ==================== Main Evaluation Pipeline ====================

class GDPvalEvaluator:
    """Main evaluator orchestrating the entire pipeline"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.task_manager = GDPvalTaskManager(config.data_dir)
        self.results = {
            "metadata": {
                "evaluation_id": self._generate_evaluation_id(),
                "timestamp": datetime.now().isoformat(),
                "config": asdict(config)
            },
            "task_results": [],
            "grading_results": [],
            "analysis": {}
        }
        
    def _generate_evaluation_id(self) -> str:
        """Generate unique evaluation ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"eval_{timestamp}_{random_suffix}"
    
    async def run_evaluation(self) -> Dict:
        """Run complete evaluation pipeline"""
        logger.info("=" * 60)
        logger.info("Starting GDPval Evaluation")
        logger.info(f"Evaluation ID: {self.results['metadata']['evaluation_id']}")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Load tasks
            logger.info("Phase 1: Loading tasks")
            tasks = self.task_manager.load_tasks(self.config.task_categories)
            
            if not tasks:
                logger.error("No tasks loaded")
                return self.results
            
            # Phase 2: Load models
            logger.info("Phase 2: Loading models")
            evaluators = await self._load_evaluators()
            
            # Phase 3: Run evaluations
            logger.info("Phase 3: Running evaluations")
            for evaluator in evaluators:
                logger.info(f"Evaluating with {evaluator.model_type}")
                
                if self.config.parallel_execution:
                    task_results = await self._evaluate_parallel(evaluator, tasks)
                else:
                    task_results = await self._evaluate_sequential(evaluator, tasks)
                
                self.results["task_results"].extend(task_results)
                
                # Save intermediate results if configured
                if self.config.save_intermediate:
                    self._save_intermediate_results(evaluator.model_type, task_results)
            
            # Phase 4: Grading
            if self.config.use_automated_grading or self.config.use_human_grading:
                logger.info("Phase 4: Grading outputs")
                grading_system = GradingSystem({
                    "use_automated": self.config.use_automated_grading,
                    "use_human": self.config.use_human_grading
                })
                
                # Group results by task
                results_by_task = defaultdict(list)
                for result in self.results["task_results"]:
                    results_by_task[result["task_id"]].append(result)
                
                # Grade each task
                for task in tasks:
                    task_id = task["task_id"]
                    if task_id in results_by_task:
                        grading_results = await grading_system.grade_outputs(
                            task,
                            results_by_task[task_id]
                        )
                        self.results["grading_results"].extend(grading_results)
            
            # Phase 5: Analysis
            logger.info("Phase 5: Analyzing results")
            self.results["analysis"] = self._analyze_results()
            
            # Phase 6: Economic Impact
            if self.config.economic_analysis:
                logger.info("Phase 6: Computing economic impact")
                self.results["economic_impact"] = self._compute_economic_impact(tasks)
            
            # Save final results
            self._save_final_results()
            
            logger.info("=" * 60)
            logger.info("Evaluation Complete")
            logger.info(f"Results saved to {self.config.output_dir}")
            logger.info("=" * 60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.results["error"] = str(e)
            self._save_final_results()
            raise
    
    async def _load_evaluators(self) -> List[ModelEvaluator]:
        """Load all model evaluators"""
        evaluators = []
        
        # Load main model
        main_evaluator = ModelEvaluator(self.config.model_path, "khmer")
        main_evaluator.load_model()
        evaluators.append(main_evaluator)
        
        # Load baseline models
        if self.config.baseline_models:
            for baseline in self.config.baseline_models:
                if baseline == "gpt-4":
                    evaluator = ModelEvaluator("gpt-4", "openai")
                elif baseline == "claude-3":
                    evaluator = ModelEvaluator("claude-3-opus-20240229", "anthropic")
                else:
                    continue
                
                evaluator.load_model()
                evaluators.append(evaluator)
        
        return evaluators
    
    async def _evaluate_parallel(self, evaluator: ModelEvaluator, tasks: List[Dict]) -> List[Dict]:
        """Evaluate tasks in parallel"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def eval_with_semaphore(task):
            async with semaphore:
                return await evaluator.evaluate_task(task)
        
        results = []
        tasks_with_progress = tqdm(tasks, desc=f"Evaluating {evaluator.model_type}")
        
        # Create tasks
        eval_tasks = [eval_with_semaphore(task) for task in tasks]
        
        # Execute in batches to show progress
        for i in range(0, len(eval_tasks), self.config.max_concurrent_tasks):
            batch = eval_tasks[i:i + self.config.max_concurrent_tasks]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    results.append({"success": False, "error": str(result)})
                else:
                    results.append(result)
            
            tasks_with_progress.update(len(batch))
        
        return results
    
    async def _evaluate_sequential(self, evaluator: ModelEvaluator, tasks: List[Dict]) -> List[Dict]:
        """Evaluate tasks sequentially"""
        results = []
        
        for task in tqdm(tasks, desc=f"Evaluating {evaluator.model_type}"):
            result = await evaluator.evaluate_task(task)
            results.append(result)
        
        return results
    
    def _analyze_results(self) -> Dict:
        """Analyze evaluation results"""
        analysis = {
            "summary_statistics": {},
            "model_comparison": {},
            "task_difficulty_analysis": {},
            "category_performance": {}
        }
        
        # Calculate success rates
        results_df = pd.DataFrame(self.results["task_results"])
        
        if not results_df.empty:
            # Overall statistics
            analysis["summary_statistics"] = {
                "total_evaluations": len(results_df),
                "successful_evaluations": results_df["success"].sum(),
                "success_rate": results_df["success"].mean(),
                "avg_execution_time": results_df[results_df["success"]]["execution_time"].mean(),
                "total_execution_time": results_df["execution_time"].sum()
            }
            
            # Per-model statistics
            for model_type in results_df["model_type"].unique():
                model_results = results_df[results_df["model_type"] == model_type]
                analysis["model_comparison"][model_type] = {
                    "success_rate": model_results["success"].mean(),
                    "avg_execution_time": model_results[model_results["success"]]["execution_time"].mean(),
                    "total_tasks": len(model_results)
                }
        
        # Grading analysis
        if self.results["grading_results"]:
            grading_df = pd.DataFrame(self.results["grading_results"])
            
            if "weighted_score" in grading_df.columns:
                analysis["grading_summary"] = {
                    "avg_score": grading_df["weighted_score"].mean(),
                    "std_score": grading_df["weighted_score"].std(),
                    "min_score": grading_df["weighted_score"].min(),
                    "max_score": grading_df["weighted_score"].max()
                }
            
            if "winner" in grading_df.columns:
                win_rates = {}
                for model in grading_df["model_a"].unique():
                    wins = len(grading_df[(grading_df["model_a"] == model) & (grading_df["winner"] == "A")])
                    wins += len(grading_df[(grading_df["model_b"] == model) & (grading_df["winner"] == "B")])
                    total = len(grading_df[(grading_df["model_a"] == model) | (grading_df["model_b"] == model)])
                    win_rates[model] = wins / total if total > 0 else 0
                
                analysis["win_rates"] = win_rates
        
        return analysis
    
    def _compute_economic_impact(self, tasks: List[Dict]) -> Dict:
        """Compute economic impact of AI assistance"""
        impact = {
            "total_time_saved_hours": 0,
            "total_cost_saved_usd": 0,
            "by_category": {},
            "roi_percentage": 0
        }
        
        for task in tasks:
            task_id = task["task_id"]
            
            # Find results for this task
            task_results = [r for r in self.results["task_results"] if r["task_id"] == task_id]
            
            if task_results:
                # Get best (fastest successful) result
                successful_results = [r for r in task_results if r["success"]]
                
                if successful_results:
                    best_result = min(successful_results, key=lambda x: x["execution_time"])
                    
                    # Calculate savings
                    human_time_hours = task["task_metadata"]["estimated_time_minutes"] / 60
                    ai_time_hours = best_result["execution_time"] / 3600
                    time_saved = human_time_hours - ai_time_hours
                    
                    hourly_wage = task["task_metadata"]["wage_per_hour_usd"]
                    cost_saved = time_saved * hourly_wage
                    
                    impact["total_time_saved_hours"] += time_saved
                    impact["total_cost_saved_usd"] += cost_saved
                    
                    # By category
                    category = task["category"]
                    if category not in impact["by_category"]:
                        impact["by_category"][category] = {
                            "time_saved_hours": 0,
                            "cost_saved_usd": 0,
                            "task_count": 0
                        }
                    
                    impact["by_category"][category]["time_saved_hours"] += time_saved
                    impact["by_category"][category]["cost_saved_usd"] += cost_saved
                    impact["by_category"][category]["task_count"] += 1
        
        # Calculate ROI
        total_human_cost = sum(
            task["task_metadata"]["estimated_time_minutes"] / 60 * task["task_metadata"]["wage_per_hour_usd"]
            for task in tasks
        )
        
        if total_human_cost > 0:
            impact["roi_percentage"] = (impact["total_cost_saved_usd"] / total_human_cost) * 100
        
        return impact
    
    def _save_intermediate_results(self, model_type: str, results: List[Dict]):
        """Save intermediate results"""
        output_file = Path(self.config.output_dir) / f"intermediate_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved intermediate results to {output_file}")
    
    def _save_final_results(self):
        """Save final evaluation results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        output_file = output_dir / f"evaluation_{self.results['metadata']['evaluation_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Save summary
        summary_file = output_dir / f"summary_{self.results['metadata']['evaluation_id']}.json"
        summary = {
            "evaluation_id": self.results["metadata"]["evaluation_id"],
            "timestamp": self.results["metadata"]["timestamp"],
            "analysis": self.results.get("analysis", {}),
            "economic_impact": self.results.get("economic_impact", {})
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report()
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")
    
    def _generate_markdown_report(self):
        """Generate markdown report of results"""
        report_file = Path(self.config.output_dir) / f"report_{self.results['metadata']['evaluation_id']}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# GDPval Evaluation Report\n\n")
            f.write(f"**Evaluation ID:** {self.results['metadata']['evaluation_id']}\n")
            f.write(f"**Timestamp:** {self.results['metadata']['timestamp']}\n\n")
            
            # Summary statistics
            if "analysis" in self.results and "summary_statistics" in self.results["analysis"]:
                stats = self.results["analysis"]["summary_statistics"]
                f.write("## Summary Statistics\n\n")
                f.write(f"- Total Evaluations: {stats.get('total_evaluations', 0)}\n")
                f.write(f"- Success Rate: {stats.get('success_rate', 0):.2%}\n")
                f.write(f"- Average Execution Time: {stats.get('avg_execution_time', 0):.2f} seconds\n\n")
            
            # Model comparison
            if "analysis" in self.results and "model_comparison" in self.results["analysis"]:
                f.write("## Model Comparison\n\n")
                f.write("| Model | Success Rate | Avg Time (s) | Tasks |\n")
                f.write("|-------|-------------|--------------|-------|\n")
                
                for model, metrics in self.results["analysis"]["model_comparison"].items():
                    f.write(f"| {model} | {metrics['success_rate']:.2%} | "
                           f"{metrics['avg_execution_time']:.2f} | {metrics['total_tasks']} |\n")
                f.write("\n")
            
            # Economic impact
            if "economic_impact" in self.results:
                impact = self.results["economic_impact"]
                f.write("## Economic Impact\n\n")
                f.write(f"- Total Time Saved: {impact['total_time_saved_hours']:.2f} hours\n")
                f.write(f"- Total Cost Saved: ${impact['total_cost_saved_usd']:.2f}\n")
                f.write(f"- ROI: {impact['roi_percentage']:.1f}%\n\n")
                
                if impact["by_category"]:
                    f.write("### Impact by Category\n\n")
                    f.write("| Category | Time Saved (hrs) | Cost Saved (USD) | Tasks |\n")
                    f.write("|----------|-----------------|------------------|-------|\n")
                    
                    for cat, metrics in impact["by_category"].items():
                        f.write(f"| {cat} | {metrics['time_saved_hours']:.2f} | "
                               f"${metrics['cost_saved_usd']:.2f} | {metrics['task_count']} |\n")
        
        logger.info(f"Report saved to {report_file}")


# ==================== CLI Interface ====================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GDPval Evaluation Framework for Khmer Language Models"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to evaluate"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tasks/gold_set",
        help="Directory containing evaluation tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluations",
        help="Directory for evaluation results"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Task categories to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--baseline-models",
        type=str,
        nargs="+",
        default=["gpt-4"],
        help="Baseline models for comparison"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run evaluations in parallel"
    )
    
    parser.add_argument(
        "--no-grading",
        action="store_true",
        help="Skip grading phase"
    )
    
    parser.add_argument(
        "--no-economic",
        action="store_true",
        help="Skip economic impact analysis"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = EvaluationConfig.from_yaml(args.config)
    else:
        config = EvaluationConfig(
            model_path=args.model_path,
            task_categories=args.categories,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_automated_grading=not args.no_grading,
            baseline_models=args.baseline_models,
            parallel_execution=args.parallel,
            economic_analysis=not args.no_economic,
            verbose=args.verbose
        )
    
    # Set logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run evaluation
    evaluator = GDPvalEvaluator(config)
    
    try:
        results = asyncio.run(evaluator.run_evaluation())
        
        # Print summary
        if "analysis" in results and "summary_statistics" in results["analysis"]:
            stats = results["analysis"]["summary_statistics"]
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            print(f"Success Rate: {stats.get('success_rate', 0):.2%}")
            print(f"Total Time: {stats.get('total_execution_time', 0):.2f} seconds")
            
            if "economic_impact" in results:
                impact = results["economic_impact"]
                print(f"Time Saved: {impact['total_time_saved_hours']:.2f} hours")
                print(f"Cost Saved: ${impact['total_cost_saved_usd']:.2f}")
                print(f"ROI: {impact['roi_percentage']:.1f}%")
            
            print(f"\nResults saved to: {config.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())