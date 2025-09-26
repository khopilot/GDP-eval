"""
Task Loader for GDPval Evaluation Framework
Handles loading and preprocessing of evaluation tasks
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTask:
    """Represents a single evaluation task"""
    task_id: str
    occupation: str
    category: str
    industry: str
    gdp_contribution_percent: float
    prompt: Dict[str, str]
    reference_files: List[Dict[str, Any]]
    grading_criteria: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return asdict(self)

    def get_hash(self) -> str:
        """Generate unique hash for task"""
        task_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()[:16]


class KhmerTaskLoader:
    """Loads and manages evaluation tasks for Khmer language models"""

    def __init__(self, data_dir: str):
        """
        Initialize task loader

        Args:
            data_dir: Directory containing task JSON files
        """
        self.data_dir = Path(data_dir)
        self.tasks: List[EvaluationTask] = []
        self.task_index: Dict[str, EvaluationTask] = {}

    def load_tasks(self, categories: Optional[List[str]] = None) -> List[EvaluationTask]:
        """
        Load evaluation tasks from JSON files

        Args:
            categories: Optional list of categories to filter by

        Returns:
            List of loaded evaluation tasks
        """
        logger.info(f"Loading tasks from {self.data_dir}")

        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []

        task_files = list(self.data_dir.glob("**/*.json"))
        all_tasks = []

        for task_file in tqdm(task_files, desc="Loading tasks"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle both single task and batch formats
                if 'tasks' in data:
                    tasks_data = data['tasks']
                elif 'task_id' in data:
                    tasks_data = [data]
                else:
                    logger.warning(f"Invalid task format in {task_file}")
                    continue

                for task_data in tasks_data:
                    task = self._parse_task(task_data)

                    # Filter by category if specified
                    if categories and task.category not in categories:
                        continue

                    all_tasks.append(task)
                    self.task_index[task.task_id] = task

            except Exception as e:
                logger.error(f"Error loading task file {task_file}: {e}")
                continue

        self.tasks = all_tasks
        logger.info(f"Loaded {len(all_tasks)} tasks from {len(task_files)} files")

        return all_tasks

    def _parse_task(self, task_data: Dict[str, Any]) -> EvaluationTask:
        """
        Parse task data into EvaluationTask object

        Args:
            task_data: Raw task data from JSON

        Returns:
            Parsed EvaluationTask object
        """
        return EvaluationTask(
            task_id=task_data.get('task_id', ''),
            occupation=task_data.get('occupation', ''),
            category=task_data.get('category', 'general'),
            industry=task_data.get('industry', ''),
            gdp_contribution_percent=task_data.get('gdp_contribution_percent', 0.0),
            prompt=task_data.get('prompt', {}),
            reference_files=task_data.get('reference_files', []),
            grading_criteria=task_data.get('grading_criteria', []),
            metadata=task_data.get('task_metadata', {})
        )

    def get_task_by_id(self, task_id: str) -> Optional[EvaluationTask]:
        """
        Get a specific task by ID

        Args:
            task_id: Task identifier

        Returns:
            Task if found, None otherwise
        """
        return self.task_index.get(task_id)

    def get_tasks_by_category(self, category: str) -> List[EvaluationTask]:
        """
        Get all tasks in a specific category

        Args:
            category: Category name

        Returns:
            List of tasks in the category
        """
        return [task for task in self.tasks if task.category == category]

    def get_categories(self) -> List[str]:
        """
        Get list of all unique categories

        Returns:
            List of category names
        """
        return list(set(task.category for task in self.tasks))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded tasks

        Returns:
            Dictionary with task statistics
        """
        if not self.tasks:
            return {"total_tasks": 0}

        categories = {}
        industries = {}
        languages = set()

        for task in self.tasks:
            # Count by category
            categories[task.category] = categories.get(task.category, 0) + 1

            # Count by industry
            industries[task.industry] = industries.get(task.industry, 0) + 1

            # Collect languages
            if 'language' in task.metadata:
                languages.add(task.metadata['language'])

        return {
            "total_tasks": len(self.tasks),
            "categories": categories,
            "industries": industries,
            "languages": list(languages),
            "avg_gdp_contribution": sum(t.gdp_contribution_percent for t in self.tasks) / len(self.tasks),
            "avg_difficulty": sum(t.metadata.get('difficulty_level', 0) for t in self.tasks) / len(self.tasks)
        }

    def validate_tasks(self) -> Dict[str, List[str]]:
        """
        Validate loaded tasks for completeness and correctness

        Returns:
            Dictionary with validation errors by task ID
        """
        errors = {}

        for task in self.tasks:
            task_errors = []

            # Check required fields
            if not task.task_id:
                task_errors.append("Missing task_id")
            if not task.prompt:
                task_errors.append("Missing prompt")
            if not task.grading_criteria:
                task_errors.append("Missing grading criteria")

            # Check prompt structure
            if task.prompt:
                if 'instruction' not in task.prompt:
                    task_errors.append("Missing prompt instruction")
                if 'instruction_english' not in task.prompt:
                    task_errors.append("Missing English translation of instruction")

            # Check metadata
            if not task.metadata:
                task_errors.append("Missing metadata")
            else:
                required_metadata = ['language', 'difficulty_level', 'estimated_time_minutes']
                for field in required_metadata:
                    if field not in task.metadata:
                        task_errors.append(f"Missing metadata field: {field}")

            if task_errors:
                errors[task.task_id] = task_errors

        if errors:
            logger.warning(f"Validation errors found in {len(errors)} tasks")
        else:
            logger.info("All tasks validated successfully")

        return errors