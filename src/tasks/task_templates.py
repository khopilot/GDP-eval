"""
Professional Task Templates
Templates for different occupational categories
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class OccupationCategory(Enum):
    """Main occupation categories"""
    FINANCE_BANKING = "finance_banking"
    AGRICULTURE = "agriculture"
    TOURISM_HOSPITALITY = "tourism_hospitality"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    TECHNOLOGY = "technology"
    RETAIL_TRADE = "retail_trade"
    CONSTRUCTION = "construction"
    TRANSPORTATION = "transportation"


@dataclass
class TaskTemplate:
    """Template for professional tasks"""
    category: OccupationCategory
    occupation: str
    typical_tasks: List[str]
    required_skills: List[str]
    common_tools: List[str]
    input_types: List[str]
    output_formats: List[str]
    evaluation_focus: List[str]
    time_range_minutes: tuple
    complexity_range: tuple


class TaskTemplateLibrary:
    """Library of task templates for different occupations"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, TaskTemplate]:
        """Initialize occupation-specific templates"""
        templates = {}

        # Finance & Banking Templates
        templates["loan_officer"] = TaskTemplate(
            category=OccupationCategory.FINANCE_BANKING,
            occupation="Loan Officer",
            typical_tasks=[
                "Assess loan applications",
                "Calculate repayment capacity",
                "Evaluate collateral",
                "Prepare risk assessment",
                "Recommend loan terms"
            ],
            required_skills=[
                "Financial analysis",
                "Risk assessment",
                "Credit evaluation",
                "Khmer/English documentation",
                "Regulatory compliance"
            ],
            common_tools=[
                "Financial statements",
                "Credit reports",
                "Business plans",
                "Collateral documents",
                "Income verification"
            ],
            input_types=[
                "PDF documents",
                "Excel spreadsheets",
                "Scanned documents",
                "Application forms",
                "Financial records"
            ],
            output_formats=[
                "Assessment report",
                "Risk score",
                "Loan recommendation",
                "Terms sheet",
                "Approval memo"
            ],
            evaluation_focus=[
                "Accuracy of analysis",
                "Risk identification",
                "Regulatory compliance",
                "Documentation quality",
                "Decision justification"
            ],
            time_range_minutes=(30, 90),
            complexity_range=(2, 4)
        )

        templates["financial_analyst"] = TaskTemplate(
            category=OccupationCategory.FINANCE_BANKING,
            occupation="Financial Analyst",
            typical_tasks=[
                "Prepare financial reports",
                "Analyze market trends",
                "Forecast revenue",
                "Evaluate investments",
                "Create budget proposals"
            ],
            required_skills=[
                "Data analysis",
                "Financial modeling",
                "Market research",
                "Report writing",
                "Presentation skills"
            ],
            common_tools=[
                "Financial data",
                "Market reports",
                "Historical trends",
                "Economic indicators",
                "Company financials"
            ],
            input_types=[
                "CSV files",
                "Financial databases",
                "Market data",
                "Reports",
                "Economic data"
            ],
            output_formats=[
                "Analysis report",
                "Excel model",
                "Presentation",
                "Dashboard",
                "Recommendations"
            ],
            evaluation_focus=[
                "Analysis depth",
                "Accuracy of calculations",
                "Insight quality",
                "Visualization clarity",
                "Actionable recommendations"
            ],
            time_range_minutes=(45, 180),
            complexity_range=(3, 5)
        )

        # Agriculture Templates
        templates["agricultural_advisor"] = TaskTemplate(
            category=OccupationCategory.AGRICULTURE,
            occupation="Agricultural Advisor",
            typical_tasks=[
                "Crop planning recommendations",
                "Pest management advice",
                "Yield forecasting",
                "Market price analysis",
                "Farmer training materials"
            ],
            required_skills=[
                "Crop knowledge",
                "Pest identification",
                "Weather analysis",
                "Market understanding",
                "Khmer communication"
            ],
            common_tools=[
                "Weather data",
                "Soil reports",
                "Market prices",
                "Crop calendars",
                "Pest databases"
            ],
            input_types=[
                "Field photos",
                "Weather data",
                "Soil test results",
                "Historical yields",
                "Market reports"
            ],
            output_formats=[
                "Advisory report",
                "Planting schedule",
                "Treatment plan",
                "Yield forecast",
                "Training guide"
            ],
            evaluation_focus=[
                "Technical accuracy",
                "Practicality",
                "Local relevance",
                "Cost-effectiveness",
                "Clarity for farmers"
            ],
            time_range_minutes=(20, 60),
            complexity_range=(2, 4)
        )

        # Tourism & Hospitality Templates
        templates["tour_coordinator"] = TaskTemplate(
            category=OccupationCategory.TOURISM_HOSPITALITY,
            occupation="Tour Coordinator",
            typical_tasks=[
                "Design tour itineraries",
                "Calculate tour costs",
                "Coordinate logistics",
                "Handle customer inquiries",
                "Prepare booking confirmations"
            ],
            required_skills=[
                "Itinerary planning",
                "Cost calculation",
                "Customer service",
                "Multilingual communication",
                "Local knowledge"
            ],
            common_tools=[
                "Booking systems",
                "Maps and routes",
                "Hotel databases",
                "Transportation schedules",
                "Activity catalogs"
            ],
            input_types=[
                "Customer requirements",
                "Budget constraints",
                "Date ranges",
                "Group sizes",
                "Special requests"
            ],
            output_formats=[
                "Itinerary document",
                "Cost breakdown",
                "Booking confirmation",
                "Travel guide",
                "Emergency contacts"
            ],
            evaluation_focus=[
                "Itinerary quality",
                "Cost accuracy",
                "Logistics feasibility",
                "Customer satisfaction",
                "Cultural sensitivity"
            ],
            time_range_minutes=(30, 90),
            complexity_range=(2, 4)
        )

        templates["hotel_manager"] = TaskTemplate(
            category=OccupationCategory.TOURISM_HOSPITALITY,
            occupation="Hotel Manager",
            typical_tasks=[
                "Room allocation planning",
                "Staff scheduling",
                "Revenue management",
                "Guest complaint resolution",
                "Service quality reports"
            ],
            required_skills=[
                "Operations management",
                "Revenue optimization",
                "Staff management",
                "Problem solving",
                "Customer relations"
            ],
            common_tools=[
                "Booking systems",
                "Staff rosters",
                "Revenue reports",
                "Guest feedback",
                "Inventory systems"
            ],
            input_types=[
                "Occupancy data",
                "Staff availability",
                "Guest complaints",
                "Revenue targets",
                "Event schedules"
            ],
            output_formats=[
                "Operations report",
                "Staff schedule",
                "Action plan",
                "Performance analysis",
                "Guest response"
            ],
            evaluation_focus=[
                "Operational efficiency",
                "Revenue optimization",
                "Problem resolution",
                "Staff utilization",
                "Guest satisfaction"
            ],
            time_range_minutes=(45, 120),
            complexity_range=(3, 5)
        )

        # Manufacturing Templates
        templates["quality_controller"] = TaskTemplate(
            category=OccupationCategory.MANUFACTURING,
            occupation="Quality Controller",
            typical_tasks=[
                "Inspect product quality",
                "Document defects",
                "Analyze quality trends",
                "Prepare quality reports",
                "Recommend improvements"
            ],
            required_skills=[
                "Quality standards",
                "Defect identification",
                "Statistical analysis",
                "Report writing",
                "Problem analysis"
            ],
            common_tools=[
                "Quality checklists",
                "Measurement tools",
                "Defect logs",
                "Statistical software",
                "Standards documents"
            ],
            input_types=[
                "Product samples",
                "Quality data",
                "Production logs",
                "Standards specs",
                "Historical data"
            ],
            output_formats=[
                "Inspection report",
                "Defect analysis",
                "Quality metrics",
                "Improvement plan",
                "Compliance certificate"
            ],
            evaluation_focus=[
                "Inspection accuracy",
                "Defect identification",
                "Root cause analysis",
                "Documentation quality",
                "Improvement suggestions"
            ],
            time_range_minutes=(30, 90),
            complexity_range=(2, 4)
        )

        templates["production_planner"] = TaskTemplate(
            category=OccupationCategory.MANUFACTURING,
            occupation="Production Planner",
            typical_tasks=[
                "Create production schedules",
                "Calculate material requirements",
                "Optimize resource allocation",
                "Monitor production progress",
                "Adjust plans for delays"
            ],
            required_skills=[
                "Production planning",
                "Resource optimization",
                "Scheduling",
                "Problem solving",
                "Data analysis"
            ],
            common_tools=[
                "Production software",
                "Inventory systems",
                "Capacity data",
                "Order backlogs",
                "Resource calendars"
            ],
            input_types=[
                "Order volumes",
                "Capacity constraints",
                "Material availability",
                "Delivery deadlines",
                "Equipment schedules"
            ],
            output_formats=[
                "Production schedule",
                "Material plan",
                "Resource allocation",
                "Progress report",
                "Adjustment recommendations"
            ],
            evaluation_focus=[
                "Schedule feasibility",
                "Resource efficiency",
                "Deadline achievement",
                "Cost optimization",
                "Flexibility planning"
            ],
            time_range_minutes=(60, 180),
            complexity_range=(3, 5)
        )

        # Healthcare Templates
        templates["healthcare_admin"] = TaskTemplate(
            category=OccupationCategory.HEALTHCARE,
            occupation="Healthcare Administrator",
            typical_tasks=[
                "Patient scheduling",
                "Resource allocation",
                "Report preparation",
                "Budget management",
                "Staff coordination"
            ],
            required_skills=[
                "Healthcare systems",
                "Resource management",
                "Data analysis",
                "Budgeting",
                "Communication"
            ],
            common_tools=[
                "Patient records",
                "Scheduling systems",
                "Budget reports",
                "Staff databases",
                "Medical supplies"
            ],
            input_types=[
                "Patient data",
                "Staff schedules",
                "Budget constraints",
                "Supply levels",
                "Department requests"
            ],
            output_formats=[
                "Schedule plan",
                "Resource report",
                "Budget analysis",
                "Staff roster",
                "Operations summary"
            ],
            evaluation_focus=[
                "Scheduling efficiency",
                "Resource optimization",
                "Budget compliance",
                "Service coverage",
                "Coordination quality"
            ],
            time_range_minutes=(30, 120),
            complexity_range=(2, 4)
        )

        # Education Templates
        templates["education_coordinator"] = TaskTemplate(
            category=OccupationCategory.EDUCATION,
            occupation="Education Coordinator",
            typical_tasks=[
                "Curriculum planning",
                "Teacher scheduling",
                "Student assessment analysis",
                "Resource allocation",
                "Parent communication"
            ],
            required_skills=[
                "Curriculum knowledge",
                "Scheduling",
                "Data analysis",
                "Communication",
                "Educational planning"
            ],
            common_tools=[
                "Curriculum guides",
                "Assessment data",
                "Teacher availability",
                "Resource inventories",
                "Communication platforms"
            ],
            input_types=[
                "Student data",
                "Teacher schedules",
                "Curriculum requirements",
                "Assessment results",
                "Parent feedback"
            ],
            output_formats=[
                "Curriculum plan",
                "Teacher schedule",
                "Assessment report",
                "Resource plan",
                "Communication memo"
            ],
            evaluation_focus=[
                "Educational alignment",
                "Schedule efficiency",
                "Resource utilization",
                "Communication clarity",
                "Student outcomes"
            ],
            time_range_minutes=(45, 120),
            complexity_range=(2, 4)
        )

        return templates

    def get_template(self, occupation: str) -> TaskTemplate:
        """Get template for specific occupation"""
        occupation_key = occupation.lower().replace(" ", "_")
        return self.templates.get(occupation_key)

    def list_occupations(self) -> List[str]:
        """List all available occupations"""
        return [template.occupation for template in self.templates.values()]

    def get_by_category(self, category: OccupationCategory) -> List[TaskTemplate]:
        """Get all templates in a category"""
        return [
            template for template in self.templates.values()
            if template.category == category
        ]

    def generate_task_prompt(
        self,
        occupation: str,
        task_type: str,
        complexity: int = 3
    ) -> Dict[str, Any]:
        """
        Generate a task prompt based on template

        Args:
            occupation: Occupation name
            task_type: Type of task
            complexity: Complexity level (1-5)

        Returns:
            Task prompt structure
        """
        template = self.get_template(occupation)
        if not template:
            raise ValueError(f"No template found for occupation: {occupation}")

        # Select appropriate elements based on complexity
        time_estimate = (
            template.time_range_minutes[0] +
            (template.time_range_minutes[1] - template.time_range_minutes[0]) *
            (complexity - 1) / 4
        )

        return {
            "occupation": template.occupation,
            "task_type": task_type,
            "complexity": complexity,
            "required_skills": template.required_skills[:min(complexity + 1, len(template.required_skills))],
            "expected_inputs": template.input_types[:min(complexity, len(template.input_types))],
            "expected_output": template.output_formats[0],
            "evaluation_criteria": template.evaluation_focus[:min(complexity + 1, len(template.evaluation_focus))],
            "estimated_time_minutes": time_estimate,
            "tools_available": template.common_tools[:min(complexity + 2, len(template.common_tools))]
        }

    def export_templates(self, output_file: str):
        """Export all templates to file"""
        import json
        export_data = {}
        for key, template in self.templates.items():
            export_data[key] = {
                "category": template.category.value,
                "occupation": template.occupation,
                "typical_tasks": template.typical_tasks,
                "required_skills": template.required_skills,
                "evaluation_focus": template.evaluation_focus,
                "complexity_range": template.complexity_range,
                "time_range_minutes": template.time_range_minutes
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)