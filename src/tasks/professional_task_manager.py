"""
Professional Task Manager
Manages collection, validation, and integration of real professional tasks
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
import yaml

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels"""
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    SPECIALIST = 5


class TaskFrequency(Enum):
    """How often task is performed"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    AD_HOC = "ad_hoc"


class ValidationStatus(Enum):
    """Task validation status"""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    EXPERT_REVIEW = "expert_review"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class TaskContributor:
    """Professional who contributed the task"""
    contributor_id: str
    name: str
    occupation: str
    organization: str
    years_experience: int
    email: Optional[str] = None
    phone: Optional[str] = None
    verified: bool = False
    contribution_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TaskInput:
    """Input required for task"""
    name: str
    file_type: str
    description: str
    required: bool = True
    sample_path: Optional[str] = None
    size_estimate: Optional[str] = None


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating task output"""
    criterion: str
    weight: float
    description: str
    success_indicators: List[str]
    failure_indicators: List[str]


@dataclass
class ProfessionalTask:
    """Professional task for evaluation"""
    # Identification
    task_id: str
    title: str
    description: str

    # Classification
    occupation: str
    sector: str
    complexity: TaskComplexity
    frequency: TaskFrequency

    # Task details
    real_world_context: str
    business_scenario: str
    inputs: List[TaskInput]
    expected_output: Dict[str, Any]

    # Time and effort
    estimated_time_minutes: float
    actual_time_minutes: Optional[float] = None

    # Contributor
    contributor: TaskContributor = None

    # Evaluation
    evaluation_criteria: List[EvaluationCriteria] = field(default_factory=list)
    quality_indicators: Dict[str, str] = field(default_factory=dict)
    common_errors: List[str] = field(default_factory=list)

    # Economic impact
    hourly_wage_usd: float = 0.0
    business_impact: str = ""
    automation_potential: float = 0.0
    economic_value_usd: float = 0.0

    # Validation
    validation_status: ValidationStatus = ValidationStatus.SUBMITTED
    validator_notes: List[str] = field(default_factory=list)
    validation_date: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert enums to values
        data['complexity'] = self.complexity.value
        data['frequency'] = self.frequency.value
        data['validation_status'] = self.validation_status.value
        return data

    def calculate_economic_value(self):
        """Calculate economic value of the task"""
        # Time value
        time_value = (self.estimated_time_minutes / 60) * self.hourly_wage_usd

        # Automation savings
        if self.automation_potential > 0:
            automation_savings = time_value * self.automation_potential
        else:
            automation_savings = 0

        self.economic_value_usd = time_value + automation_savings
        return self.economic_value_usd


class ProfessionalTaskManager:
    """Manages professional tasks"""

    def __init__(self, base_path: str = "data/professional_tasks"):
        """
        Initialize task manager

        Args:
            base_path: Base directory for tasks
        """
        self.base_path = Path(base_path)
        self.templates_dir = self.base_path / "templates"
        self.submitted_dir = self.base_path / "submitted"
        self.validated_dir = self.base_path / "validated"
        self.metadata_dir = self.base_path / "metadata"

        # Create directories if they don't exist
        for dir_path in [self.templates_dir, self.submitted_dir,
                         self.validated_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.occupations = self._load_occupations()
        self.contributors = self._load_contributors()

    def _load_occupations(self) -> Dict[str, Any]:
        """Load occupation metadata"""
        occupation_file = self.metadata_dir / "occupations.json"
        if occupation_file.exists():
            with open(occupation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_contributors(self) -> Dict[str, TaskContributor]:
        """Load contributor database"""
        contributor_file = self.metadata_dir / "contributors.json"
        if contributor_file.exists():
            with open(contributor_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    cid: TaskContributor(**cdata)
                    for cid, cdata in data.items()
                }
        return {}

    def submit_task(
        self,
        task: ProfessionalTask,
        contributor: TaskContributor
    ) -> str:
        """
        Submit a new professional task

        Args:
            task: Professional task
            contributor: Task contributor

        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if not task.task_id:
            task.task_id = self._generate_task_id(task.sector, task.occupation)

        # Add contributor
        task.contributor = contributor

        # Save contributor if new
        if contributor.contributor_id not in self.contributors:
            self.contributors[contributor.contributor_id] = contributor
            self._save_contributors()

        # Calculate economic value
        task.calculate_economic_value()

        # Save task to submitted directory
        task_file = self.submitted_dir / f"{task.task_id}.yaml"
        with open(task_file, 'w', encoding='utf-8') as f:
            yaml.dump(task.to_dict(), f, allow_unicode=True)

        logger.info(f"Task submitted: {task.task_id}")
        return task.task_id

    def validate_task(
        self,
        task_id: str,
        validator_id: str,
        approved: bool,
        notes: Optional[str] = None
    ) -> bool:
        """
        Validate a submitted task

        Args:
            task_id: Task ID
            validator_id: Validator ID
            approved: Whether task is approved
            notes: Validation notes

        Returns:
            Success status
        """
        # Load task
        task = self.load_task(task_id, status="submitted")
        if not task:
            logger.error(f"Task not found: {task_id}")
            return False

        # Update validation status
        if approved:
            task.validation_status = ValidationStatus.VALIDATED
            task.validation_date = datetime.now().isoformat()

            # Move to validated directory
            source = self.submitted_dir / f"{task_id}.yaml"
            dest = self.validated_dir / f"{task_id}.yaml"

            # Save validated task
            with open(dest, 'w', encoding='utf-8') as f:
                yaml.dump(task.to_dict(), f, allow_unicode=True)

            # Remove from submitted
            if source.exists():
                source.unlink()

            logger.info(f"Task validated: {task_id}")
        else:
            task.validation_status = ValidationStatus.NEEDS_REVISION

            # Save with notes
            task_file = self.submitted_dir / f"{task_id}.yaml"
            with open(task_file, 'w', encoding='utf-8') as f:
                yaml.dump(task.to_dict(), f, allow_unicode=True)

            logger.info(f"Task needs revision: {task_id}")

        # Add validator notes
        if notes:
            task.validator_notes.append(f"[{validator_id}] {notes}")

        return True

    def load_task(
        self,
        task_id: str,
        status: str = "validated"
    ) -> Optional[ProfessionalTask]:
        """
        Load a task by ID

        Args:
            task_id: Task ID
            status: Task status (submitted/validated)

        Returns:
            Professional task or None
        """
        # Determine directory
        if status == "validated":
            task_dir = self.validated_dir
        else:
            task_dir = self.submitted_dir

        # Load task file
        task_file = task_dir / f"{task_id}.yaml"
        if not task_file.exists():
            return None

        with open(task_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Convert to ProfessionalTask
        # Handle enums
        if 'complexity' in data:
            data['complexity'] = TaskComplexity(data['complexity'])
        if 'frequency' in data:
            data['frequency'] = TaskFrequency(data['frequency'])
        if 'validation_status' in data:
            data['validation_status'] = ValidationStatus(data['validation_status'])

        # Handle nested objects
        if 'contributor' in data and data['contributor']:
            data['contributor'] = TaskContributor(**data['contributor'])

        if 'inputs' in data:
            data['inputs'] = [TaskInput(**inp) for inp in data['inputs']]

        if 'evaluation_criteria' in data:
            data['evaluation_criteria'] = [
                EvaluationCriteria(**crit) for crit in data['evaluation_criteria']
            ]

        return ProfessionalTask(**data)

    def list_tasks(
        self,
        status: str = "validated",
        sector: Optional[str] = None,
        occupation: Optional[str] = None
    ) -> List[ProfessionalTask]:
        """
        List tasks with optional filters

        Args:
            status: Task status
            sector: Filter by sector
            occupation: Filter by occupation

        Returns:
            List of tasks
        """
        # Determine directory
        if status == "validated":
            task_dir = self.validated_dir
        else:
            task_dir = self.submitted_dir

        tasks = []
        for task_file in task_dir.glob("*.yaml"):
            task = self.load_task(task_file.stem, status)
            if task:
                # Apply filters
                if sector and task.sector != sector:
                    continue
                if occupation and task.occupation != occupation:
                    continue
                tasks.append(task)

        return tasks

    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about professional tasks"""
        validated_tasks = self.list_tasks(status="validated")
        submitted_tasks = self.list_tasks(status="submitted")

        # Sector distribution
        sector_dist = {}
        for task in validated_tasks:
            sector_dist[task.sector] = sector_dist.get(task.sector, 0) + 1

        # Complexity distribution
        complexity_dist = {}
        for task in validated_tasks:
            level = task.complexity.name
            complexity_dist[level] = complexity_dist.get(level, 0) + 1

        # Economic value
        total_value = sum(task.economic_value_usd for task in validated_tasks)
        avg_value = total_value / len(validated_tasks) if validated_tasks else 0

        return {
            "total_validated": len(validated_tasks),
            "total_submitted": len(submitted_tasks),
            "sectors": sector_dist,
            "complexity": complexity_dist,
            "total_economic_value": total_value,
            "average_economic_value": avg_value,
            "contributors": len(self.contributors),
            "average_time_minutes": sum(
                t.estimated_time_minutes for t in validated_tasks
            ) / len(validated_tasks) if validated_tasks else 0
        }

    def _generate_task_id(self, sector: str, occupation: str) -> str:
        """Generate unique task ID"""
        sector_abbr = sector[:3].lower()
        occupation_abbr = occupation.replace(" ", "_")[:10].lower()
        unique_id = uuid.uuid4().hex[:8]
        return f"{sector_abbr}_{occupation_abbr}_{unique_id}"

    def _save_contributors(self):
        """Save contributor database"""
        contributor_file = self.metadata_dir / "contributors.json"
        data = {
            cid: asdict(contributor)
            for cid, contributor in self.contributors.items()
        }
        with open(contributor_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_for_evaluation(
        self,
        output_file: str,
        include_submitted: bool = False
    ):
        """
        Export tasks for evaluation pipeline

        Args:
            output_file: Output file path
            include_submitted: Include non-validated tasks
        """
        tasks = self.list_tasks(status="validated")
        if include_submitted:
            tasks.extend(self.list_tasks(status="submitted"))

        # Convert to evaluation format
        export_data = []
        for task in tasks:
            export_data.append({
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "sector": task.sector,
                "complexity": task.complexity.value,
                "inputs": [asdict(inp) for inp in task.inputs],
                "expected_output": task.expected_output,
                "evaluation_criteria": [
                    asdict(crit) for crit in task.evaluation_criteria
                ],
                "economic_value": task.economic_value_usd
            })

        # Save to file
        output_path = Path(output_file)
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, allow_unicode=True)

        logger.info(f"Exported {len(export_data)} tasks to {output_file}")