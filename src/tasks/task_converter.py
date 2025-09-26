"""
Task Converter
Bridges professional tasks with evaluation pipeline
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from src.core.task_loader import EvaluationTask
from src.tasks.professional_task_manager import ProfessionalTask, ProfessionalTaskManager
from src.tasks.task_templates import TaskTemplateLibrary
from src.tasks.occupation_mapper import OccupationMapper

logger = logging.getLogger(__name__)


class TaskConverter:
    """
    Converts between professional tasks and evaluation tasks
    """

    def __init__(self):
        """Initialize task converter"""
        self.task_manager = ProfessionalTaskManager()
        self.templates = TaskTemplateLibrary()
        self.occupation_mapper = OccupationMapper()

    def professional_to_evaluation(
        self,
        professional_task: ProfessionalTask
    ) -> EvaluationTask:
        """
        Convert professional task to evaluation task

        Args:
            professional_task: Professional task object

        Returns:
            Evaluation task
        """
        # Build prompt from professional task
        prompt = self._build_prompt(professional_task)

        # Extract grading criteria
        grading_criteria = []
        for criteria in professional_task.evaluation_criteria:
            # Handle both dict and object formats
            if isinstance(criteria, dict):
                # Already in dict format from our fix
                grading_criteria.append({
                    "criterion_id": criteria.get("criterion_id", "general_quality"),
                    "criterion_name": criteria.get("criterion_id", "General Quality").replace("_", " ").title(),
                    "weight": criteria.get("weight", 0.25),
                    "max_score": criteria.get("max_score", 10.0),
                    "description": criteria.get("description", "Quality assessment"),
                    "success_indicators": criteria.get("success_indicators", []),
                    "failure_indicators": criteria.get("failure_indicators", [])
                })
            else:
                # Original object format
                grading_criteria.append({
                    "criterion_id": criteria.criterion.lower().replace(" ", "_"),
                    "criterion_name": criteria.criterion,
                    "weight": criteria.weight,
                    "max_score": 10.0,
                    "description": criteria.description,
                    "success_indicators": criteria.success_indicators,
                    "failure_indicators": criteria.failure_indicators
                })

        # Build reference files if inputs provided
        reference_files = []
        for input_spec in professional_task.inputs:
            if input_spec.sample_path:
                reference_files.append({
                    "file_name": input_spec.name,
                    "file_type": input_spec.file_type,
                    "file_path": input_spec.sample_path,
                    "description": input_spec.description
                })

        # Get GDP contribution for sector
        gdp_contributions = {
            "finance": 6.0,
            "agriculture": 22.0,
            "tourism": 12.0,
            "manufacturing": 18.0,
            "healthcare": 1.0,
            "education": 1.0,
            "technology": 3.0,
            "retail_trade": 13.0,
            "construction": 9.0,
            "transportation": 6.0
        }

        # Create evaluation task
        eval_task = EvaluationTask(
            task_id=professional_task.task_id,
            category=professional_task.sector,
            occupation=professional_task.occupation,
            industry=professional_task.sector,  # Use sector as industry
            gdp_contribution_percent=gdp_contributions.get(professional_task.sector, 5.0),
            prompt=prompt,
            reference_files=reference_files if reference_files else [],
            grading_criteria=grading_criteria if grading_criteria else [
                {
                    "criterion_id": "general_quality",
                    "criterion_name": "General Quality",
                    "weight": 1.0,
                    "max_score": 10.0
                }
            ],
            metadata={
                "language": self._detect_language(professional_task),
                "complexity": professional_task.complexity.value if hasattr(professional_task.complexity, 'value') else professional_task.complexity,
                "estimated_time_minutes": professional_task.estimated_time_minutes,
                "hourly_wage_usd": professional_task.hourly_wage_usd,
                "business_impact": professional_task.business_impact,
                "automation_potential": professional_task.automation_potential,
                "economic_value_usd": professional_task.economic_value_usd,
                "frequency": professional_task.frequency.value if hasattr(professional_task.frequency, 'value') else professional_task.frequency
            }
        )

        return eval_task

    def _build_prompt(self, task: ProfessionalTask) -> Dict[str, Any]:
        """
        Build evaluation prompt from professional task

        Args:
            task: Professional task

        Returns:
            Prompt dictionary
        """
        prompt = {
            "instruction": task.description,
            "context": f"{task.real_world_context}\n\nBusiness Scenario: {task.business_scenario}",
            "requirements": [],
            "output_format": task.expected_output.get("format", "structured response")
        }

        # Add expected outputs as requirements
        if "deliverables" in task.expected_output:
            for deliverable in task.expected_output["deliverables"]:
                prompt["requirements"].append(deliverable)

        # Add quality indicators as requirements
        if task.quality_indicators:
            for indicator, description in task.quality_indicators.items():
                prompt["requirements"].append(f"{indicator}: {description}")

        # Add language requirement
        language = self._detect_language(task)
        if language == "khmer":
            prompt["language_note"] = "សូមឆ្លើយជាភាសាខ្មែរ (Please respond in Khmer)"
        elif language == "bilingual":
            prompt["language_note"] = "Please provide bilingual response (Khmer/English)"

        return prompt

    def _detect_language(self, task: ProfessionalTask) -> str:
        """
        Detect language requirement from task

        Args:
            task: Professional task

        Returns:
            Language code (khmer, english, bilingual)
        """
        # Check task tags
        if task.tags:
            if "khmer" in task.tags:
                return "khmer"
            if "bilingual" in task.tags:
                return "bilingual"

        # Check occupation
        occupation_mapping = self.occupation_mapper.map_occupation(task.occupation)
        if occupation_mapping and occupation_mapping.khmer_title:
            return "bilingual"

        return "english"

    def create_cambodia_test_suite(self) -> List[EvaluationTask]:
        """
        Create comprehensive test suite with Cambodia professional tasks

        Returns:
            List of evaluation tasks
        """
        test_tasks = []

        # 1. Finance - Loan Officer
        loan_task = ProfessionalTask(
            task_id="finance_loan_001",
            title="SME Loan Assessment",
            description="Evaluate loan application for small restaurant business in Phnom Penh",
            occupation="Loan Officer",
            sector="finance",
            complexity=3,
            frequency="daily",
            real_world_context="A family-owned restaurant in BKK1 area has been operating for 3 years. They're applying for expansion loan to open second location.",
            business_scenario="Current monthly revenue: $12,000. Expenses: $8,000. Requesting: $50,000 loan. Collateral: Restaurant equipment ($30,000) and personal property ($40,000).",
            inputs=[],
            expected_output={
                "format": "Loan assessment report",
                "deliverables": [
                    "Risk level assessment (low/medium/high)",
                    "Recommended interest rate",
                    "Suggested repayment period",
                    "Collateral evaluation",
                    "Final recommendation (approve/reject/modify)"
                ]
            },
            estimated_time_minutes=45,
            hourly_wage_usd=6.25,
            business_impact="Enables business expansion and job creation",
            automation_potential=0.45,
            economic_value_usd=4.69,
            evaluation_criteria=[
                {"criterion_id": "accuracy", "weight": 0.3, "max_score": 10},
                {"criterion_id": "completeness", "weight": 0.3, "max_score": 10},
                {"criterion_id": "technical_correctness", "weight": 0.2, "max_score": 10},
                {"criterion_id": "language_quality", "weight": 0.2, "max_score": 10}
            ],
            quality_indicators={
                "accuracy": "Correct risk assessment based on financials",
                "completeness": "All required elements addressed",
                "compliance": "Follows NBC regulations"
            },
            tags=["bilingual", "financial", "sme"]
        )
        test_tasks.append(self.professional_to_evaluation(loan_task))

        # 2. Agriculture - Rice Farming Advisor
        agri_task = ProfessionalTask(
            task_id="agri_rice_001",
            title="Rice Planting Advisory",
            description="ផ្តល់ដំបូន្មានអំពីការដាំស្រូវរដូវវស្សា (Provide rainy season rice planting advice)",
            occupation="Agricultural Advisor",
            sector="agriculture",
            complexity=3,
            frequency="weekly",
            real_world_context="Farmer in Battambang province with 3 hectares of land. Previous season affected by drought. Has access to small irrigation canal.",
            business_scenario="Soil type: Clay loam. Previous crop: Cassava. Available budget: $500 for inputs. Labor: Family of 4 plus occasional hired help.",
            inputs=[],
            expected_output={
                "format": "Agricultural advisory report",
                "deliverables": [
                    "Recommended rice varieties (2-3 options)",
                    "Planting calendar with key dates",
                    "Fertilizer application schedule",
                    "Water management plan",
                    "Pest management strategy",
                    "Expected yield range"
                ]
            },
            estimated_time_minutes=30,
            hourly_wage_usd=5.00,
            business_impact="Improves crop yield and farmer income",
            automation_potential=0.35,
            economic_value_usd=2.50,
            evaluation_criteria=[
                {"criterion_id": "accuracy", "weight": 0.3, "max_score": 10},
                {"criterion_id": "completeness", "weight": 0.3, "max_score": 10},
                {"criterion_id": "technical_correctness", "weight": 0.2, "max_score": 10},
                {"criterion_id": "language_quality", "weight": 0.2, "max_score": 10}
            ],
            quality_indicators={
                "local_relevance": "Suitable for Battambang conditions",
                "practicality": "Feasible with available resources",
                "technical_accuracy": "Agronomically sound advice"
            },
            tags=["khmer", "agriculture", "rural"]
        )
        test_tasks.append(self.professional_to_evaluation(agri_task))

        # 3. Tourism - Tour Package Design
        tour_task = ProfessionalTask(
            task_id="tourism_angkor_001",
            title="Angkor Temple Tour Package",
            description="Design 2-day tour package for Korean family visiting Siem Reap",
            occupation="Tour Coordinator",
            sector="tourism",
            complexity=2,
            frequency="daily",
            real_world_context="Korean family of 4 (parents age 45, children age 12 and 15). First time in Cambodia. Interested in history and photography.",
            business_scenario="Staying at hotel near Pub Street. Budget: $600 total for tours. Dates: December weekend. Prefer Korean-speaking guide if available.",
            inputs=[],
            expected_output={
                "format": "Tour itinerary with pricing",
                "deliverables": [
                    "Day-by-day detailed itinerary",
                    "Temple pass and entrance fees breakdown",
                    "Transportation arrangements",
                    "Guide services and language options",
                    "Meal recommendations with options",
                    "Total cost calculation",
                    "Special photography spots and best times"
                ]
            },
            estimated_time_minutes=40,
            hourly_wage_usd=4.50,
            business_impact="Enhances tourist experience and repeat visits",
            automation_potential=0.40,
            economic_value_usd=3.00,
            evaluation_criteria=[
                {"criterion_id": "accuracy", "weight": 0.3, "max_score": 10},
                {"criterion_id": "completeness", "weight": 0.3, "max_score": 10},
                {"criterion_id": "technical_correctness", "weight": 0.2, "max_score": 10},
                {"criterion_id": "language_quality", "weight": 0.2, "max_score": 10}
            ],
            quality_indicators={
                "cultural_sensitivity": "Appropriate for Korean visitors",
                "value_for_money": "Good balance of cost and experience",
                "feasibility": "Realistic timing and logistics"
            },
            tags=["english", "tourism", "customer_service"]
        )
        test_tasks.append(self.professional_to_evaluation(tour_task))

        # 4. Manufacturing - Quality Control
        qc_task = ProfessionalTask(
            task_id="mfg_garment_001",
            title="Garment Quality Inspection",
            description="Inspect batch of t-shirts for export quality standards",
            occupation="Quality Controller",
            sector="manufacturing",
            complexity=2,
            frequency="daily",
            real_world_context="Garment factory producing for US brand. Batch of 500 t-shirts ready for inspection. Previous batch had 3% defect rate.",
            business_scenario="Order specifications: 100% cotton, specific color (Pantone 19-4052), size tolerance ±2cm, straight seams, no loose threads. Deadline: Ship tomorrow.",
            inputs=[],
            expected_output={
                "format": "Quality inspection report",
                "deliverables": [
                    "Sampling plan (how many to inspect)",
                    "Defect categories to check",
                    "Inspection results summary",
                    "Defect rate calculation",
                    "Pass/fail decision",
                    "Corrective actions if needed"
                ]
            },
            estimated_time_minutes=60,
            hourly_wage_usd=3.50,
            business_impact="Prevents costly returns and maintains buyer relationships",
            automation_potential=0.65,
            economic_value_usd=3.50,
            evaluation_criteria=[
                {"criterion_id": "accuracy", "weight": 0.3, "max_score": 10},
                {"criterion_id": "completeness", "weight": 0.3, "max_score": 10},
                {"criterion_id": "technical_correctness", "weight": 0.2, "max_score": 10},
                {"criterion_id": "language_quality", "weight": 0.2, "max_score": 10}
            ],
            quality_indicators={
                "thoroughness": "Comprehensive inspection coverage",
                "accuracy": "Correct defect identification",
                "efficiency": "Appropriate sampling method"
            },
            tags=["english", "manufacturing", "quality"]
        )
        test_tasks.append(self.professional_to_evaluation(qc_task))

        # 5. Healthcare - Patient Triage
        health_task = ProfessionalTask(
            task_id="health_triage_001",
            title="Rural Clinic Patient Triage",
            description="Prioritize patients at busy rural health center",
            occupation="Medical Assistant",
            sector="healthcare",
            complexity=3,
            frequency="daily",
            real_world_context="Rural health center in Kampong Thom. Limited resources: 1 doctor, 2 nurses. 15 patients waiting.",
            business_scenario="Patients include: child with high fever, elderly with chest pain, pregnant woman for check-up, motorcycle accident victim with leg injury, several with common cold symptoms.",
            inputs=[],
            expected_output={
                "format": "Triage priority list with reasoning",
                "deliverables": [
                    "Priority order of patients (1-5 scale)",
                    "Reasoning for each priority level",
                    "Immediate actions for critical cases",
                    "Estimated wait times",
                    "Resource allocation plan"
                ]
            },
            estimated_time_minutes=20,
            hourly_wage_usd=4.00,
            business_impact="Saves lives and optimizes limited healthcare resources",
            automation_potential=0.20,
            economic_value_usd=1.33,
            evaluation_criteria=[
                {"criterion_id": "accuracy", "weight": 0.3, "max_score": 10},
                {"criterion_id": "completeness", "weight": 0.3, "max_score": 10},
                {"criterion_id": "technical_correctness", "weight": 0.2, "max_score": 10},
                {"criterion_id": "language_quality", "weight": 0.2, "max_score": 10}
            ],
            quality_indicators={
                "medical_accuracy": "Correct urgency assessment",
                "practicality": "Feasible with available resources",
                "clarity": "Clear communication for staff"
            },
            tags=["bilingual", "healthcare", "emergency"]
        )
        test_tasks.append(self.professional_to_evaluation(health_task))

        logger.info(f"Created {len(test_tasks)} professional test tasks for Cambodia context")
        return test_tasks

    def load_validated_tasks(
        self,
        limit: Optional[int] = None
    ) -> List[EvaluationTask]:
        """
        Load validated professional tasks from storage

        Args:
            limit: Maximum number of tasks to load

        Returns:
            List of evaluation tasks
        """
        professional_tasks = self.task_manager.list_tasks(status="validated")

        if limit:
            professional_tasks = professional_tasks[:limit]

        eval_tasks = []
        for prof_task in professional_tasks:
            try:
                eval_task = self.professional_to_evaluation(prof_task)
                eval_tasks.append(eval_task)
            except Exception as e:
                logger.error(f"Failed to convert task {prof_task.task_id}: {e}")

        logger.info(f"Loaded {len(eval_tasks)} validated professional tasks")
        return eval_tasks