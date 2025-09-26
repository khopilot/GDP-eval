"""
Sector-Specific Economic Evaluator
Evaluates AI/ML impact for specific economic sectors in Cambodia
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EconomicSector(Enum):
    """Economic sectors in Cambodia"""
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    TOURISM = "tourism"
    CONSTRUCTION = "construction"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    RETAIL = "retail"
    LOGISTICS = "logistics"
    ENERGY = "energy"


@dataclass
class SectorProfile:
    """Profile of an economic sector"""
    name: str
    gdp_contribution_percent: float
    employment_thousands: int
    digitalization_level: float  # 0-1 scale
    ai_readiness: float  # 0-1 scale
    growth_rate_annual: float
    export_oriented: bool
    key_challenges: List[str]
    automation_potential: float  # 0-1 scale


class SectorEvaluator:
    """
    Evaluates sector-specific economic impact of AI adoption
    """

    def __init__(self):
        """Initialize sector evaluator with Cambodia-specific data"""
        self.sector_profiles = self._initialize_sector_profiles()
        self.sector_use_cases = self._initialize_use_cases()

    def _initialize_sector_profiles(self) -> Dict[str, SectorProfile]:
        """Initialize sector profiles with Cambodia data"""
        return {
            EconomicSector.AGRICULTURE.value: SectorProfile(
                name="Agriculture",
                gdp_contribution_percent=22.0,
                employment_thousands=2800,
                digitalization_level=0.15,
                ai_readiness=0.10,
                growth_rate_annual=1.8,
                export_oriented=True,
                key_challenges=["Low tech adoption", "Climate vulnerability", "Market access"],
                automation_potential=0.35
            ),
            EconomicSector.MANUFACTURING.value: SectorProfile(
                name="Manufacturing",
                gdp_contribution_percent=18.0,
                employment_thousands=1100,
                digitalization_level=0.35,
                ai_readiness=0.25,
                growth_rate_annual=7.1,
                export_oriented=True,
                key_challenges=["Quality control", "Supply chain", "Skills gap"],
                automation_potential=0.65
            ),
            EconomicSector.SERVICES.value: SectorProfile(
                name="Services",
                gdp_contribution_percent=30.0,
                employment_thousands=1500,
                digitalization_level=0.45,
                ai_readiness=0.35,
                growth_rate_annual=6.5,
                export_oriented=False,
                key_challenges=["Customer service", "Efficiency", "Digital transformation"],
                automation_potential=0.55
            ),
            EconomicSector.TOURISM.value: SectorProfile(
                name="Tourism",
                gdp_contribution_percent=12.0,
                employment_thousands=630,
                digitalization_level=0.40,
                ai_readiness=0.30,
                growth_rate_annual=5.5,
                export_oriented=True,
                key_challenges=["Seasonality", "Marketing", "Service quality"],
                automation_potential=0.40
            ),
            EconomicSector.FINANCE.value: SectorProfile(
                name="Finance & Banking",
                gdp_contribution_percent=6.0,
                employment_thousands=85,
                digitalization_level=0.65,
                ai_readiness=0.55,
                growth_rate_annual=8.5,
                export_oriented=False,
                key_challenges=["Risk assessment", "Fraud detection", "Financial inclusion"],
                automation_potential=0.75
            ),
            EconomicSector.HEALTHCARE.value: SectorProfile(
                name="Healthcare",
                gdp_contribution_percent=1.0,
                employment_thousands=120,
                digitalization_level=0.25,
                ai_readiness=0.20,
                growth_rate_annual=9.2,
                export_oriented=False,
                key_challenges=["Diagnosis accuracy", "Resource allocation", "Rural access"],
                automation_potential=0.45
            ),
            EconomicSector.EDUCATION.value: SectorProfile(
                name="Education",
                gdp_contribution_percent=1.0,
                employment_thousands=180,
                digitalization_level=0.30,
                ai_readiness=0.25,
                growth_rate_annual=5.8,
                export_oriented=False,
                key_challenges=["Quality", "Access", "Personalization"],
                automation_potential=0.50
            ),
        }

    def _initialize_use_cases(self) -> Dict[str, List[Dict]]:
        """Initialize AI use cases by sector"""
        return {
            EconomicSector.AGRICULTURE.value: [
                {
                    "use_case": "Crop yield prediction",
                    "impact": 0.15,
                    "implementation_cost": 50000,
                    "time_to_implement_months": 6
                },
                {
                    "use_case": "Pest detection",
                    "impact": 0.10,
                    "implementation_cost": 30000,
                    "time_to_implement_months": 4
                },
                {
                    "use_case": "Market price forecasting",
                    "impact": 0.12,
                    "implementation_cost": 25000,
                    "time_to_implement_months": 3
                }
            ],
            EconomicSector.FINANCE.value: [
                {
                    "use_case": "Credit risk assessment",
                    "impact": 0.25,
                    "implementation_cost": 150000,
                    "time_to_implement_months": 8
                },
                {
                    "use_case": "Fraud detection",
                    "impact": 0.20,
                    "implementation_cost": 100000,
                    "time_to_implement_months": 6
                },
                {
                    "use_case": "Customer service automation",
                    "impact": 0.18,
                    "implementation_cost": 75000,
                    "time_to_implement_months": 4
                }
            ],
            EconomicSector.MANUFACTURING.value: [
                {
                    "use_case": "Quality control automation",
                    "impact": 0.20,
                    "implementation_cost": 200000,
                    "time_to_implement_months": 10
                },
                {
                    "use_case": "Predictive maintenance",
                    "impact": 0.15,
                    "implementation_cost": 120000,
                    "time_to_implement_months": 6
                },
                {
                    "use_case": "Supply chain optimization",
                    "impact": 0.18,
                    "implementation_cost": 80000,
                    "time_to_implement_months": 5
                }
            ],
            EconomicSector.HEALTHCARE.value: [
                {
                    "use_case": "Diagnostic assistance",
                    "impact": 0.22,
                    "implementation_cost": 100000,
                    "time_to_implement_months": 8
                },
                {
                    "use_case": "Patient triage",
                    "impact": 0.15,
                    "implementation_cost": 50000,
                    "time_to_implement_months": 4
                },
                {
                    "use_case": "Drug interaction checking",
                    "impact": 0.12,
                    "implementation_cost": 30000,
                    "time_to_implement_months": 3
                }
            ]
        }

    def evaluate_sector(
        self,
        sector: str,
        ai_capability_score: float,
        implementation_budget: float
    ) -> Dict[str, any]:
        """
        Evaluate AI impact for a specific sector

        Args:
            sector: Economic sector name
            ai_capability_score: AI model capability (0-1)
            implementation_budget: Available budget in USD

        Returns:
            Sector evaluation results
        """
        if sector not in self.sector_profiles:
            raise ValueError(f"Unknown sector: {sector}")

        profile = self.sector_profiles[sector]
        use_cases = self.sector_use_cases.get(sector, [])

        # Calculate feasible use cases within budget
        feasible_cases = [
            case for case in use_cases
            if case['implementation_cost'] <= implementation_budget
        ]

        # Calculate potential impact
        readiness_factor = profile.ai_readiness
        digitalization_factor = profile.digitalization_level
        capability_factor = ai_capability_score

        # Combined effectiveness score
        effectiveness = (
            readiness_factor * 0.3 +
            digitalization_factor * 0.3 +
            capability_factor * 0.4
        )

        # Calculate economic impact
        if feasible_cases:
            total_impact = sum(case['impact'] for case in feasible_cases)
            avg_implementation_time = sum(
                case['time_to_implement_months'] for case in feasible_cases
            ) / len(feasible_cases)
        else:
            total_impact = 0
            avg_implementation_time = 0

        # Adjusted impact based on effectiveness
        adjusted_impact = total_impact * effectiveness

        # Calculate specific metrics
        productivity_gain = adjusted_impact * profile.automation_potential
        cost_reduction = productivity_gain * 0.6  # 60% of productivity converts to cost savings
        quality_improvement = effectiveness * 0.25  # 25% quality boost

        # Employment impact
        jobs_displaced = int(profile.employment_thousands * productivity_gain * 0.1)
        jobs_created = int(jobs_displaced * 0.4)  # 40% new jobs created
        net_employment_change = jobs_created - jobs_displaced

        evaluation = {
            'sector': sector,
            'profile': {
                'name': profile.name,
                'gdp_contribution': profile.gdp_contribution_percent,
                'employment': profile.employment_thousands,
                'digitalization': profile.digitalization_level,
                'ai_readiness': profile.ai_readiness
            },
            'feasibility': {
                'feasible_use_cases': len(feasible_cases),
                'total_use_cases': len(use_cases),
                'implementation_time_months': avg_implementation_time,
                'budget_utilization': sum(
                    case['implementation_cost'] for case in feasible_cases
                ) / implementation_budget if implementation_budget > 0 else 0
            },
            'impact': {
                'productivity_gain_percent': productivity_gain * 100,
                'cost_reduction_percent': cost_reduction * 100,
                'quality_improvement_percent': quality_improvement * 100,
                'effectiveness_score': effectiveness,
                'adjusted_impact_score': adjusted_impact
            },
            'employment': {
                'jobs_at_risk': jobs_displaced,
                'new_jobs_created': jobs_created,
                'net_change': net_employment_change,
                'reskilling_needed': jobs_displaced
            },
            'recommendations': self._generate_recommendations(
                profile, effectiveness, feasible_cases
            )
        }

        return evaluation

    def _generate_recommendations(
        self,
        profile: SectorProfile,
        effectiveness: float,
        feasible_cases: List[Dict]
    ) -> List[str]:
        """Generate sector-specific recommendations"""
        recommendations = []

        # Readiness recommendations
        if profile.ai_readiness < 0.3:
            recommendations.append(
                "Invest in basic digital infrastructure and training"
            )
        elif profile.ai_readiness < 0.6:
            recommendations.append(
                "Focus on pilot projects to build AI capabilities"
            )
        else:
            recommendations.append(
                "Scale successful AI implementations across operations"
            )

        # Digitalization recommendations
        if profile.digitalization_level < 0.3:
            recommendations.append(
                "Prioritize digitalization of core business processes"
            )

        # Use case recommendations
        if feasible_cases:
            top_case = max(feasible_cases, key=lambda x: x['impact'])
            recommendations.append(
                f"Start with '{top_case['use_case']}' for maximum impact"
            )

        # Sector-specific challenges
        for challenge in profile.key_challenges[:2]:
            recommendations.append(
                f"Address '{challenge}' through targeted AI solutions"
            )

        # Employment recommendations
        if profile.automation_potential > 0.5:
            recommendations.append(
                "Develop comprehensive reskilling programs for workforce"
            )

        return recommendations

    def compare_sectors(
        self,
        sectors: List[str],
        ai_capability_score: float,
        budget_per_sector: float
    ) -> Dict[str, any]:
        """
        Compare multiple sectors for AI investment prioritization

        Args:
            sectors: List of sectors to compare
            ai_capability_score: AI capability level
            budget_per_sector: Budget available per sector

        Returns:
            Comparative analysis
        """
        evaluations = {}
        for sector in sectors:
            if sector in self.sector_profiles:
                evaluations[sector] = self.evaluate_sector(
                    sector, ai_capability_score, budget_per_sector
                )

        # Rank sectors by impact
        rankings = sorted(
            evaluations.items(),
            key=lambda x: x[1]['impact']['adjusted_impact_score'],
            reverse=True
        )

        comparison = {
            'rankings': [
                {
                    'rank': i + 1,
                    'sector': sector,
                    'impact_score': eval['impact']['adjusted_impact_score'],
                    'productivity_gain': eval['impact']['productivity_gain_percent']
                }
                for i, (sector, eval) in enumerate(rankings)
            ],
            'best_sector': rankings[0][0] if rankings else None,
            'evaluations': evaluations
        }

        return comparison

    def export_evaluation(
        self,
        evaluation: Dict,
        filepath: str
    ):
        """Export evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(evaluation, f, indent=2)
        logger.info(f"Evaluation exported to {filepath}")