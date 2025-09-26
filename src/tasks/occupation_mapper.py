"""
Occupation Mapper for Cambodia Labor Market
Maps between international occupation codes and local categories
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ISCOLevel(Enum):
    """International Standard Classification of Occupations levels"""
    MAJOR_GROUP = 1  # 1 digit
    SUB_MAJOR = 2    # 2 digits
    MINOR_GROUP = 3  # 3 digits
    UNIT_GROUP = 4   # 4 digits


@dataclass
class OccupationMapping:
    """Mapping between occupation classifications"""
    isco_code: str
    isco_title: str
    khmer_title: str
    local_code: str
    sector: str
    skill_level: int  # 1-4 (ISCO skill levels)
    average_wage_usd: float
    workforce_percentage: float
    automation_risk: float  # 0-1


class OccupationMapper:
    """
    Maps occupations between different classification systems
    Focused on Cambodian labor market
    """

    def __init__(self):
        """Initialize occupation mapper"""
        self.mappings = self._initialize_mappings()
        self.sector_map = self._initialize_sectors()
        self.skill_map = self._initialize_skills()

    def _initialize_mappings(self) -> Dict[str, OccupationMapping]:
        """Initialize Cambodia-specific occupation mappings"""
        mappings = {}

        # Agriculture sector mappings
        mappings["6111"] = OccupationMapping(
            isco_code="6111",
            isco_title="Field crop and vegetable growers",
            khmer_title="អ្នកដាំដំណាំស្រែ និងបន្លែ",
            local_code="AG-01",
            sector="agriculture",
            skill_level=2,
            average_wage_usd=150,
            workforce_percentage=5.2,
            automation_risk=0.3
        )

        mappings["6112"] = OccupationMapping(
            isco_code="6112",
            isco_title="Tree and shrub crop growers",
            khmer_title="អ្នកដាំដើមឈើ និងដំណាំព្រៃឈើ",
            local_code="AG-02",
            sector="agriculture",
            skill_level=2,
            average_wage_usd=160,
            workforce_percentage=3.8,
            automation_risk=0.25
        )

        mappings["6121"] = OccupationMapping(
            isco_code="6121",
            isco_title="Livestock and dairy producers",
            khmer_title="អ្នកចិញ្ចឹមសត្វ និងផលិតទឹកដោះគោ",
            local_code="AG-03",
            sector="agriculture",
            skill_level=2,
            average_wage_usd=180,
            workforce_percentage=2.1,
            automation_risk=0.35
        )

        # Manufacturing sector mappings
        mappings["7531"] = OccupationMapping(
            isco_code="7531",
            isco_title="Tailors, dressmakers, furriers",
            khmer_title="ជាងកាត់ដេរសម្លៀកបំពាក់",
            local_code="MF-01",
            sector="manufacturing",
            skill_level=2,
            average_wage_usd=200,
            workforce_percentage=8.5,
            automation_risk=0.65
        )

        mappings["8152"] = OccupationMapping(
            isco_code="8152",
            isco_title="Weaving and knitting machine operators",
            khmer_title="អ្នកប្រតិបត្តិម៉ាស៊ីនត្បាញ",
            local_code="MF-02",
            sector="manufacturing",
            skill_level=2,
            average_wage_usd=190,
            workforce_percentage=6.3,
            automation_risk=0.75
        )

        # Tourism & hospitality mappings
        mappings["5111"] = OccupationMapping(
            isco_code="5111",
            isco_title="Travel attendants and travel stewards",
            khmer_title="បុគ្គលិកទេសចរណ៍",
            local_code="TH-01",
            sector="tourism_hospitality",
            skill_level=2,
            average_wage_usd=250,
            workforce_percentage=1.2,
            automation_risk=0.4
        )

        mappings["5120"] = OccupationMapping(
            isco_code="5120",
            isco_title="Cooks",
            khmer_title="មេចុងភៅ",
            local_code="TH-02",
            sector="tourism_hospitality",
            skill_level=2,
            average_wage_usd=220,
            workforce_percentage=2.8,
            automation_risk=0.35
        )

        mappings["1411"] = OccupationMapping(
            isco_code="1411",
            isco_title="Hotel managers",
            khmer_title="អ្នកគ្រប់គ្រងសណ្ឋាគារ",
            local_code="TH-03",
            sector="tourism_hospitality",
            skill_level=4,
            average_wage_usd=800,
            workforce_percentage=0.3,
            automation_risk=0.15
        )

        # Finance & banking mappings
        mappings["3312"] = OccupationMapping(
            isco_code="3312",
            isco_title="Credit and loans officers",
            khmer_title="មន្ត្រីឥណទាន និងប្រាក់កម្ចី",
            local_code="FB-01",
            sector="finance_banking",
            skill_level=3,
            average_wage_usd=400,
            workforce_percentage=0.8,
            automation_risk=0.45
        )

        mappings["2411"] = OccupationMapping(
            isco_code="2411",
            isco_title="Accountants",
            khmer_title="គណនេយ្យករ",
            local_code="FB-02",
            sector="finance_banking",
            skill_level=4,
            average_wage_usd=500,
            workforce_percentage=0.6,
            automation_risk=0.55
        )

        mappings["4211"] = OccupationMapping(
            isco_code="4211",
            isco_title="Bank tellers and related clerks",
            khmer_title="បុគ្គលិកធនាគារ",
            local_code="FB-03",
            sector="finance_banking",
            skill_level=2,
            average_wage_usd=300,
            workforce_percentage=1.1,
            automation_risk=0.85
        )

        # Healthcare mappings
        mappings["2221"] = OccupationMapping(
            isco_code="2221",
            isco_title="Nursing professionals",
            khmer_title="គិលានុបដ្ឋាក",
            local_code="HC-01",
            sector="healthcare",
            skill_level=4,
            average_wage_usd=350,
            workforce_percentage=0.9,
            automation_risk=0.1
        )

        mappings["3256"] = OccupationMapping(
            isco_code="3256",
            isco_title="Medical assistants",
            khmer_title="ជំនួយការពេទ្យ",
            local_code="HC-02",
            sector="healthcare",
            skill_level=3,
            average_wage_usd=250,
            workforce_percentage=0.7,
            automation_risk=0.2
        )

        # Education mappings
        mappings["2341"] = OccupationMapping(
            isco_code="2341",
            isco_title="Primary school teachers",
            khmer_title="គ្រូបង្រៀនបឋមសិក្សា",
            local_code="ED-01",
            sector="education",
            skill_level=4,
            average_wage_usd=280,
            workforce_percentage=2.3,
            automation_risk=0.05
        )

        mappings["2342"] = OccupationMapping(
            isco_code="2342",
            isco_title="Early childhood educators",
            khmer_title="គ្រូមត្តេយ្យសិក្សា",
            local_code="ED-02",
            sector="education",
            skill_level=4,
            average_wage_usd=200,
            workforce_percentage=0.8,
            automation_risk=0.02
        )

        # Technology mappings
        mappings["2511"] = OccupationMapping(
            isco_code="2511",
            isco_title="Systems analysts",
            khmer_title="អ្នកវិភាគប្រព័ន្ធ",
            local_code="IT-01",
            sector="technology",
            skill_level=4,
            average_wage_usd=600,
            workforce_percentage=0.2,
            automation_risk=0.25
        )

        mappings["2514"] = OccupationMapping(
            isco_code="2514",
            isco_title="Applications programmers",
            khmer_title="អ្នកសរសេរកម្មវិធី",
            local_code="IT-02",
            sector="technology",
            skill_level=4,
            average_wage_usd=700,
            workforce_percentage=0.3,
            automation_risk=0.3
        )

        # Retail & trade mappings
        mappings["5221"] = OccupationMapping(
            isco_code="5221",
            isco_title="Shopkeepers",
            khmer_title="ម្ចាស់ហាង",
            local_code="RT-01",
            sector="retail_trade",
            skill_level=2,
            average_wage_usd=350,
            workforce_percentage=4.5,
            automation_risk=0.4
        )

        mappings["5223"] = OccupationMapping(
            isco_code="5223",
            isco_title="Shop sales assistants",
            khmer_title="អ្នកលក់ក្នុងហាង",
            local_code="RT-02",
            sector="retail_trade",
            skill_level=2,
            average_wage_usd=180,
            workforce_percentage=5.8,
            automation_risk=0.65
        )

        # Construction mappings
        mappings["7111"] = OccupationMapping(
            isco_code="7111",
            isco_title="House builders",
            khmer_title="ជាងសំណង់ផ្ទះ",
            local_code="CO-01",
            sector="construction",
            skill_level=2,
            average_wage_usd=250,
            workforce_percentage=3.2,
            automation_risk=0.35
        )

        mappings["7112"] = OccupationMapping(
            isco_code="7112",
            isco_title="Bricklayers and related workers",
            khmer_title="ជាងដាក់ឥដ្ឋ",
            local_code="CO-02",
            sector="construction",
            skill_level=2,
            average_wage_usd=200,
            workforce_percentage=2.8,
            automation_risk=0.4
        )

        # Transportation mappings
        mappings["8322"] = OccupationMapping(
            isco_code="8322",
            isco_title="Car, taxi and van drivers",
            khmer_title="អ្នកបើកតាក់ស៊ី",
            local_code="TR-01",
            sector="transportation",
            skill_level=2,
            average_wage_usd=220,
            workforce_percentage=2.5,
            automation_risk=0.7
        )

        mappings["9331"] = OccupationMapping(
            isco_code="9331",
            isco_title="Hand and pedal vehicle drivers",
            khmer_title="អ្នកបើកម៉ូតូឌុប",
            local_code="TR-02",
            sector="transportation",
            skill_level=1,
            average_wage_usd=150,
            workforce_percentage=3.1,
            automation_risk=0.5
        )

        return mappings

    def _initialize_sectors(self) -> Dict[str, Dict[str, any]]:
        """Initialize sector information for Cambodia"""
        return {
            "agriculture": {
                "khmer_name": "កសិកម្ម",
                "gdp_contribution": 0.22,
                "workforce_percentage": 31.8,
                "avg_wage_usd": 165,
                "growth_rate": 0.02,
                "formality_rate": 0.05
            },
            "manufacturing": {
                "khmer_name": "ឧស្សាហកម្មកែច្នៃ",
                "gdp_contribution": 0.18,
                "workforce_percentage": 21.5,
                "avg_wage_usd": 195,
                "growth_rate": 0.07,
                "formality_rate": 0.85
            },
            "tourism_hospitality": {
                "khmer_name": "ទេសចរណ៍ និងសេវាកម្ម",
                "gdp_contribution": 0.12,
                "workforce_percentage": 8.3,
                "avg_wage_usd": 320,
                "growth_rate": 0.15,
                "formality_rate": 0.65
            },
            "finance_banking": {
                "khmer_name": "ហិរញ្ញវត្ថុ និងធនាគារ",
                "gdp_contribution": 0.08,
                "workforce_percentage": 2.5,
                "avg_wage_usd": 450,
                "growth_rate": 0.12,
                "formality_rate": 0.95
            },
            "healthcare": {
                "khmer_name": "សុខាភិបាល",
                "gdp_contribution": 0.04,
                "workforce_percentage": 1.8,
                "avg_wage_usd": 300,
                "growth_rate": 0.08,
                "formality_rate": 0.75
            },
            "education": {
                "khmer_name": "អប់រំ",
                "gdp_contribution": 0.05,
                "workforce_percentage": 3.9,
                "avg_wage_usd": 240,
                "growth_rate": 0.05,
                "formality_rate": 0.90
            },
            "technology": {
                "khmer_name": "បច្ចេកវិទ្យា",
                "gdp_contribution": 0.03,
                "workforce_percentage": 0.5,
                "avg_wage_usd": 650,
                "growth_rate": 0.25,
                "formality_rate": 0.80
            },
            "retail_trade": {
                "khmer_name": "ពាណិជ្ជកម្ម",
                "gdp_contribution": 0.13,
                "workforce_percentage": 15.2,
                "avg_wage_usd": 265,
                "growth_rate": 0.06,
                "formality_rate": 0.40
            },
            "construction": {
                "khmer_name": "សំណង់",
                "gdp_contribution": 0.09,
                "workforce_percentage": 9.0,
                "avg_wage_usd": 225,
                "growth_rate": 0.10,
                "formality_rate": 0.35
            },
            "transportation": {
                "khmer_name": "ដឹកជញ្ជូន",
                "gdp_contribution": 0.06,
                "workforce_percentage": 5.7,
                "avg_wage_usd": 185,
                "growth_rate": 0.08,
                "formality_rate": 0.30
            }
        }

    def _initialize_skills(self) -> Dict[int, Dict[str, any]]:
        """Initialize ISCO skill level definitions"""
        return {
            1: {
                "description": "Elementary occupations",
                "education": "Primary education",
                "khmer_desc": "មុខរបរសាមញ្ញ",
                "training_months": 0
            },
            2: {
                "description": "Skilled manual work",
                "education": "Lower secondary",
                "khmer_desc": "ការងារជំនាញដោយដៃ",
                "training_months": 3
            },
            3: {
                "description": "Technical and associate",
                "education": "Upper secondary",
                "khmer_desc": "បច្ចេកទេស និងជំនួយការ",
                "training_months": 12
            },
            4: {
                "description": "Professional",
                "education": "Tertiary/University",
                "khmer_desc": "វិជ្ជាជីវៈ",
                "training_months": 36
            }
        }

    def map_occupation(
        self,
        title: str,
        sector: Optional[str] = None
    ) -> Optional[OccupationMapping]:
        """
        Map occupation title to classification

        Args:
            title: Occupation title (English or Khmer)
            sector: Optional sector hint

        Returns:
            Occupation mapping or None
        """
        title_lower = title.lower()

        # Try exact match first
        for mapping in self.mappings.values():
            if (title_lower == mapping.isco_title.lower() or
                title == mapping.khmer_title):
                return mapping

        # Try partial match
        for mapping in self.mappings.values():
            if (title_lower in mapping.isco_title.lower() or
                (mapping.khmer_title and title in mapping.khmer_title)):
                if not sector or mapping.sector == sector:
                    return mapping

        return None

    def get_sector_occupations(self, sector: str) -> List[OccupationMapping]:
        """Get all occupations in a sector"""
        return [
            mapping for mapping in self.mappings.values()
            if mapping.sector == sector
        ]

    def get_skill_level_occupations(self, skill_level: int) -> List[OccupationMapping]:
        """Get occupations by skill level"""
        return [
            mapping for mapping in self.mappings.values()
            if mapping.skill_level == skill_level
        ]

    def calculate_automation_impact(
        self,
        occupation: str,
        years_ahead: int = 5
    ) -> Dict[str, float]:
        """
        Calculate automation impact on occupation

        Args:
            occupation: Occupation title or code
            years_ahead: Years to project

        Returns:
            Impact metrics
        """
        mapping = self.map_occupation(occupation)
        if not mapping:
            return {}

        # Simple projection model
        annual_automation_rate = 0.05  # 5% per year
        cumulative_risk = min(
            1.0,
            mapping.automation_risk * (1 + annual_automation_rate * years_ahead)
        )

        jobs_at_risk = mapping.workforce_percentage * cumulative_risk
        wage_impact = mapping.average_wage_usd * jobs_at_risk * 0.01  # Percentage to decimal

        return {
            "occupation": mapping.isco_title,
            "current_risk": mapping.automation_risk,
            "projected_risk": cumulative_risk,
            "workforce_affected_percent": jobs_at_risk,
            "wage_impact_usd": wage_impact,
            "reskilling_priority": "High" if cumulative_risk > 0.7 else "Medium" if cumulative_risk > 0.4 else "Low"
        }

    def get_reskilling_pathways(
        self,
        from_occupation: str
    ) -> List[Tuple[OccupationMapping, float]]:
        """
        Suggest reskilling pathways from occupation

        Args:
            from_occupation: Current occupation

        Returns:
            List of (target_occupation, similarity_score)
        """
        source = self.map_occupation(from_occupation)
        if not source:
            return []

        pathways = []

        for target in self.mappings.values():
            # Skip same occupation
            if target.isco_code == source.isco_code:
                continue

            # Calculate similarity score
            score = 0.0

            # Same sector bonus
            if target.sector == source.sector:
                score += 0.3

            # Adjacent skill level bonus
            if abs(target.skill_level - source.skill_level) <= 1:
                score += 0.3

            # Higher wage bonus
            if target.average_wage_usd > source.average_wage_usd:
                wage_ratio = target.average_wage_usd / source.average_wage_usd
                score += min(0.2, wage_ratio * 0.1)

            # Lower automation risk bonus
            if target.automation_risk < source.automation_risk:
                risk_reduction = source.automation_risk - target.automation_risk
                score += min(0.2, risk_reduction)

            if score > 0.3:  # Minimum threshold
                pathways.append((target, score))

        # Sort by score
        pathways.sort(key=lambda x: x[1], reverse=True)
        return pathways[:5]  # Top 5 pathways

    def export_mapping_data(self, filepath: str):
        """Export occupation mappings to JSON"""
        import json

        export_data = {
            "mappings": {
                code: {
                    "isco_code": m.isco_code,
                    "isco_title": m.isco_title,
                    "khmer_title": m.khmer_title,
                    "sector": m.sector,
                    "skill_level": m.skill_level,
                    "average_wage_usd": m.average_wage_usd,
                    "automation_risk": m.automation_risk
                }
                for code, m in self.mappings.items()
            },
            "sectors": self.sector_map,
            "skill_levels": self.skill_map
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported occupation mappings to {filepath}")