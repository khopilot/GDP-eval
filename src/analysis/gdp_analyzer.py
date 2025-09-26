"""
GDP Impact Analyzer
Calculates real economic impact in Cambodian context
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class GDPMetrics:
    """GDP impact metrics"""
    gdp_impact_usd: float
    gdp_impact_percent: float
    gdp_impact_basis_points: float
    sector_contribution: Dict[str, float]
    annual_growth_contribution: float
    jobs_created: int
    productivity_multiplier: float


class GDPAnalyzer:
    """
    Analyzes GDP impact of AI/ML adoption in Cambodia
    """

    # Cambodia GDP data (2024 estimates in million USD)
    TOTAL_GDP = 32000  # Million USD

    SECTOR_GDP = {
        'agriculture': 7040,     # 22% of GDP
        'manufacturing': 5760,   # 18% of GDP
        'services': 9600,       # 30% of GDP
        'tourism': 3840,        # 12% of GDP
        'construction': 2560,   # 8% of GDP
        'finance': 1920,        # 6% of GDP
        'technology': 640,      # 2% of GDP
        'healthcare': 320,      # 1% of GDP
        'education': 320,       # 1% of GDP
    }

    # Sector productivity multipliers
    PRODUCTIVITY_MULTIPLIERS = {
        'agriculture': 1.15,      # Lower tech adoption
        'manufacturing': 1.35,    # Moderate automation potential
        'services': 1.45,        # High digitalization impact
        'tourism': 1.25,         # Service enhancement
        'construction': 1.20,    # Project management improvement
        'finance': 1.60,         # High automation potential
        'technology': 1.80,      # Highest multiplier
        'healthcare': 1.30,      # Diagnostic improvement
        'education': 1.40,       # Learning enhancement
    }

    # Employment by sector (thousands)
    SECTOR_EMPLOYMENT = {
        'agriculture': 2800,
        'manufacturing': 1100,
        'services': 1500,
        'tourism': 630,
        'construction': 380,
        'finance': 85,
        'technology': 35,
        'healthcare': 120,
        'education': 180,
    }

    def __init__(self, exchange_rate: float = 4100):
        """
        Initialize GDP analyzer

        Args:
            exchange_rate: USD to KHR exchange rate
        """
        self.exchange_rate = exchange_rate

    def calculate_gdp_impact(
        self,
        sector: str,
        productivity_gain: float,
        adoption_rate: float = 0.1,
        time_horizon_years: int = 1
    ) -> GDPMetrics:
        """
        Calculate GDP impact of technology adoption

        Args:
            sector: Economic sector
            productivity_gain: Productivity improvement (0-1)
            adoption_rate: Technology adoption rate (0-1)
            time_horizon_years: Time horizon for impact

        Returns:
            GDP impact metrics
        """
        if sector not in self.SECTOR_GDP:
            raise ValueError(f"Unknown sector: {sector}")

        # Base sector GDP
        sector_gdp = self.SECTOR_GDP[sector]

        # Calculate productivity-adjusted impact
        multiplier = self.PRODUCTIVITY_MULTIPLIERS[sector]
        effective_gain = productivity_gain * multiplier * adoption_rate

        # GDP impact in millions USD
        gdp_impact = sector_gdp * effective_gain * time_horizon_years

        # As percentage of total GDP
        gdp_percent = (gdp_impact / self.TOTAL_GDP) * 100

        # In basis points (1 bp = 0.01%)
        gdp_basis_points = gdp_percent * 100

        # Sector contribution breakdown
        sector_contribution = {
            'direct': gdp_impact * 0.6,
            'indirect': gdp_impact * 0.25,
            'induced': gdp_impact * 0.15
        }

        # Annual growth contribution
        annual_growth = gdp_impact / time_horizon_years

        # Job creation estimate
        labor_productivity = sector_gdp / self.SECTOR_EMPLOYMENT[sector]
        jobs_created = int((gdp_impact / labor_productivity) * 0.3)  # 30% new jobs

        return GDPMetrics(
            gdp_impact_usd=gdp_impact * 1_000_000,  # Convert to USD
            gdp_impact_percent=gdp_percent,
            gdp_impact_basis_points=gdp_basis_points,
            sector_contribution=sector_contribution,
            annual_growth_contribution=annual_growth,
            jobs_created=jobs_created,
            productivity_multiplier=multiplier
        )

    def calculate_multi_sector_impact(
        self,
        sector_impacts: Dict[str, float],
        adoption_rates: Optional[Dict[str, float]] = None,
        time_horizon_years: int = 1
    ) -> Dict[str, any]:
        """
        Calculate combined GDP impact across multiple sectors

        Args:
            sector_impacts: Productivity gains by sector
            adoption_rates: Adoption rates by sector
            time_horizon_years: Time horizon

        Returns:
            Combined impact metrics
        """
        if adoption_rates is None:
            adoption_rates = {s: 0.1 for s in sector_impacts}

        results = {}
        total_impact = 0
        total_jobs = 0

        for sector, productivity in sector_impacts.items():
            adoption = adoption_rates.get(sector, 0.1)
            metrics = self.calculate_gdp_impact(
                sector, productivity, adoption, time_horizon_years
            )
            results[sector] = metrics
            total_impact += metrics.gdp_impact_usd
            total_jobs += metrics.jobs_created

        # Calculate aggregate metrics
        aggregate = {
            'total_gdp_impact_usd': total_impact,
            'total_gdp_impact_percent': (total_impact / (self.TOTAL_GDP * 1_000_000)) * 100,
            'total_jobs_created': total_jobs,
            'sectors_affected': len(sector_impacts),
            'average_productivity_gain': np.mean(list(sector_impacts.values())),
            'by_sector': results
        }

        return aggregate

    def estimate_technology_roi(
        self,
        implementation_cost: float,
        gdp_metrics: GDPMetrics,
        time_horizon_years: int = 3
    ) -> Dict[str, float]:
        """
        Estimate ROI of technology implementation

        Args:
            implementation_cost: Cost in USD
            gdp_metrics: GDP impact metrics
            time_horizon_years: ROI calculation period

        Returns:
            ROI metrics
        """
        # Total benefit over time horizon
        total_benefit = gdp_metrics.gdp_impact_usd * time_horizon_years

        # Net present value (assuming 7% discount rate)
        discount_rate = 0.07
        npv = sum([
            gdp_metrics.gdp_impact_usd / ((1 + discount_rate) ** year)
            for year in range(1, time_horizon_years + 1)
        ]) - implementation_cost

        # ROI percentage
        roi = ((total_benefit - implementation_cost) / implementation_cost) * 100

        # Payback period in months
        monthly_benefit = gdp_metrics.gdp_impact_usd / 12
        payback_months = implementation_cost / monthly_benefit if monthly_benefit > 0 else float('inf')

        # Benefit-cost ratio
        bcr = total_benefit / implementation_cost if implementation_cost > 0 else float('inf')

        return {
            'roi_percent': roi,
            'npv_usd': npv,
            'payback_months': payback_months,
            'benefit_cost_ratio': bcr,
            'annual_return': gdp_metrics.gdp_impact_usd,
            'break_even_year': payback_months / 12
        }

    def calculate_regional_impact(
        self,
        gdp_metrics: GDPMetrics,
        regions: List[str] = None
    ) -> Dict[str, float]:
        """
        Distribute GDP impact across regions

        Args:
            gdp_metrics: GDP impact metrics
            regions: List of regions (default: all provinces)

        Returns:
            Regional impact distribution
        """
        # Regional GDP distribution (simplified)
        regional_weights = {
            'Phnom Penh': 0.35,
            'Siem Reap': 0.12,
            'Battambang': 0.08,
            'Kampong Cham': 0.06,
            'Kandal': 0.05,
            'Other': 0.34
        }

        if regions:
            # Filter to specified regions
            weights = {r: regional_weights.get(r, 0.05) for r in regions}
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {r: w/total_weight for r, w in weights.items()}
        else:
            weights = regional_weights

        regional_impact = {}
        for region, weight in weights.items():
            regional_impact[region] = {
                'gdp_impact_usd': gdp_metrics.gdp_impact_usd * weight,
                'jobs_created': int(gdp_metrics.jobs_created * weight),
                'share_percent': weight * 100
            }

        return regional_impact

    def export_analysis(
        self,
        gdp_metrics: GDPMetrics,
        filepath: str
    ):
        """
        Export analysis to JSON file

        Args:
            gdp_metrics: GDP metrics to export
            filepath: Output file path
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(gdp_metrics),
            'exchange_rate': self.exchange_rate,
            'gdp_baseline': self.TOTAL_GDP
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Analysis exported to {filepath}")


def main():
    """Example usage"""
    analyzer = GDPAnalyzer()

    # Single sector analysis
    metrics = analyzer.calculate_gdp_impact(
        sector='finance',
        productivity_gain=0.25,  # 25% productivity improvement
        adoption_rate=0.15,       # 15% adoption rate
        time_horizon_years=1
    )

    print(f"GDP Impact: ${metrics.gdp_impact_usd:,.0f}")
    print(f"GDP Growth: {metrics.gdp_impact_percent:.3f}%")
    print(f"Jobs Created: {metrics.jobs_created:,}")

    # Multi-sector analysis
    sector_impacts = {
        'finance': 0.30,
        'services': 0.20,
        'manufacturing': 0.15,
        'agriculture': 0.10
    }

    combined = analyzer.calculate_multi_sector_impact(sector_impacts)
    print(f"\nTotal Impact: ${combined['total_gdp_impact_usd']:,.0f}")
    print(f"Total Jobs: {combined['total_jobs_created']:,}")


if __name__ == "__main__":
    main()