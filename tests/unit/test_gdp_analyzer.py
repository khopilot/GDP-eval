"""
Unit tests for GDP Analyzer
"""

import pytest
import json
from pathlib import Path

from src.analysis.gdp_analyzer import GDPAnalyzer, GDPMetrics


class TestGDPAnalyzer:
    """Test GDP analyzer functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return GDPAnalyzer(exchange_rate=4100)

    @pytest.fixture
    def economic_data(self):
        """Load economic test data"""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "economic_data.json"
        with open(fixture_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def test_calculate_gdp_impact_single_sector(self, analyzer):
        """Test GDP impact calculation for single sector"""
        result = analyzer.calculate_gdp_impact(
            sector='finance',
            productivity_gain=0.25,
            adoption_rate=0.15,
            time_horizon_years=1
        )

        assert isinstance(result, GDPMetrics)
        assert result.gdp_impact_usd > 0
        assert 0 <= result.gdp_impact_percent <= 100
        assert result.gdp_impact_basis_points == result.gdp_impact_percent * 100
        assert result.jobs_created >= 0

    def test_calculate_multi_sector_impact(self, analyzer):
        """Test multi-sector impact calculation"""
        sector_impacts = {
            'finance': 0.30,
            'services': 0.20,
            'manufacturing': 0.15
        }

        result = analyzer.calculate_multi_sector_impact(sector_impacts)

        assert 'total_gdp_impact_usd' in result
        assert 'total_jobs_created' in result
        assert 'by_sector' in result
        assert len(result['by_sector']) == len(sector_impacts)

    def test_invalid_sector(self, analyzer):
        """Test handling of invalid sector"""
        with pytest.raises(ValueError, match="Unknown sector"):
            analyzer.calculate_gdp_impact(
                sector='invalid_sector',
                productivity_gain=0.20,
                adoption_rate=0.10
            )

    def test_estimate_technology_roi(self, analyzer):
        """Test ROI estimation"""
        gdp_metrics = GDPMetrics(
            gdp_impact_usd=1000000,
            gdp_impact_percent=0.05,
            gdp_impact_basis_points=5,
            sector_contribution={'direct': 600000},
            annual_growth_contribution=1000000,
            jobs_created=50,
            productivity_multiplier=1.5
        )

        roi = analyzer.estimate_technology_roi(
            implementation_cost=200000,
            gdp_metrics=gdp_metrics,
            time_horizon_years=3
        )

        assert 'roi_percent' in roi
        assert 'npv_usd' in roi
        assert 'payback_months' in roi
        assert roi['roi_percent'] > 0

    def test_calculate_regional_impact(self, analyzer):
        """Test regional impact distribution"""
        gdp_metrics = GDPMetrics(
            gdp_impact_usd=1000000,
            gdp_impact_percent=0.05,
            gdp_impact_basis_points=5,
            sector_contribution={'direct': 600000},
            annual_growth_contribution=1000000,
            jobs_created=100,
            productivity_multiplier=1.5
        )

        regional = analyzer.calculate_regional_impact(gdp_metrics)

        assert 'Phnom Penh' in regional
        assert 'Siem Reap' in regional
        total_impact = sum(r['gdp_impact_usd'] for r in regional.values())
        assert abs(total_impact - gdp_metrics.gdp_impact_usd) < 100  # Small rounding error ok

    def test_sector_productivity_multipliers(self, analyzer):
        """Test that sectors have different productivity multipliers"""
        tech_result = analyzer.calculate_gdp_impact(
            sector='technology',
            productivity_gain=0.20,
            adoption_rate=0.10
        )

        agri_result = analyzer.calculate_gdp_impact(
            sector='agriculture',
            productivity_gain=0.20,
            adoption_rate=0.10
        )

        # Technology should have higher multiplier than agriculture
        assert tech_result.productivity_multiplier > agri_result.productivity_multiplier

    def test_export_analysis(self, analyzer, tmp_path):
        """Test export functionality"""
        gdp_metrics = GDPMetrics(
            gdp_impact_usd=1000000,
            gdp_impact_percent=0.05,
            gdp_impact_basis_points=5,
            sector_contribution={'direct': 600000},
            annual_growth_contribution=1000000,
            jobs_created=50,
            productivity_multiplier=1.5
        )

        filepath = tmp_path / "test_analysis.json"
        analyzer.export_analysis(gdp_metrics, str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
            assert 'metrics' in data
            assert 'exchange_rate' in data

    def test_zero_adoption_rate(self, analyzer):
        """Test with zero adoption rate"""
        result = analyzer.calculate_gdp_impact(
            sector='finance',
            productivity_gain=0.25,
            adoption_rate=0.0
        )

        assert result.gdp_impact_usd == 0
        assert result.jobs_created == 0

    def test_high_impact_scenario(self, analyzer):
        """Test high impact scenario"""
        result = analyzer.calculate_gdp_impact(
            sector='technology',
            productivity_gain=0.50,  # 50% productivity gain
            adoption_rate=0.30,      # 30% adoption
            time_horizon_years=3
        )

        # Should show significant impact
        assert result.gdp_impact_usd > 0
        assert result.gdp_impact_percent > 0
        assert result.jobs_created > 0