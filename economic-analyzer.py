"""
Economic Impact Analyzer for GDPval Khmer Evaluation
Calculates real economic impact in Cambodian context with sector-specific analysis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EconomicMetrics:
    """Economic metrics for a task or sector"""
    time_saved_hours: float
    cost_saved_usd: float
    cost_saved_khr: float
    productivity_gain_percent: float
    roi_percentage: float
    break_even_months: float
    annual_impact_usd: float
    gdp_impact_basis_points: float
    jobs_affected: int
    sector: str
    

class CambodianEconomicAnalyzer:
    """
    Analyzes economic impact of AI adoption in Cambodian context
    """
    
    # Exchange rate (update regularly)
    USD_TO_KHR = 4100
    
    # Cambodia GDP data (2024 estimates in million USD)
    SECTOR_GDP = {
        'agriculture': 4500,  # ~22% of GDP
        'manufacturing': 3700,  # ~18% of GDP
        'services': 6200,  # ~30% of GDP
        'tourism': 2500,  # ~12% of GDP
        'finance': 1650,  # ~8% of GDP
        'technology': 820,  # ~4% of GDP
        'healthcare': 620,  # ~3% of GDP
        'education': 410,  # ~2% of GDP
        'government': 205,  # ~1% of GDP
    }
    
    # Average wages by occupation in Cambodia (USD/month)
    OCCUPATION_WAGES = {
        # Finance sector
        'financial_analyst': 800,
        'accountant': 600,
        'bank_teller': 350,
        'loan_officer': 500,
        
        # Technology sector
        'software_developer': 1200,
        'it_support': 400,
        'data_analyst': 700,
        'system_administrator': 600,
        
        # Healthcare sector
        'doctor': 1000,
        'nurse': 350,
        'pharmacist': 500,
        'medical_technician': 400,
        
        # Agriculture sector
        'agricultural_extension_officer': 400,
        'farm_manager': 500,
        'agricultural_technician': 350,
        
        # Tourism sector
        'hotel_manager': 700,
        'tour_guide': 300,
        'travel_agent': 400,
        
        # Education sector
        'teacher': 350,
        'university_lecturer': 600,
        'education_administrator': 500,
        
        # General
        'administrative_assistant': 350,
        'customer_service': 300,
        'sales_representative': 400,
    }
    
    # Working hours per month in Cambodia
    WORKING_HOURS_PER_MONTH = 208  # 26 days * 8 hours
    
    # AI implementation costs (estimated)
    AI_IMPLEMENTATION_COSTS = {
        'small_business': 5000,  # USD
        'medium_business': 25000,
        'large_enterprise': 100000,
        'government': 50000,
    }
    
    def __init__(self, wage_data_path: Optional[str] = None):
        """
        Initialize analyzer with optional custom wage data
        
        Args:
            wage_data_path: Path to CSV with custom wage data
        """
        self.wage_data = self.OCCUPATION_WAGES.copy()
        
        if wage_data_path and Path(wage_data_path).exists():
            self._load_custom_wages(wage_data_path)
        
        self.exchange_rate = self.USD_TO_KHR
        self._update_exchange_rate()
    
    def _load_custom_wages(self, wage_data_path: str):
        """Load custom wage data from CSV"""
        try:
            df = pd.read_csv(wage_data_path)
            for _, row in df.iterrows():
                occupation = row['occupation'].lower().replace(' ', '_')
                wage_usd = row['monthly_wage_usd']
                self.wage_data[occupation] = wage_usd
            logger.info(f"Loaded {len(df)} wage entries from {wage_data_path}")
        except Exception as e:
            logger.error(f"Error loading wage data: {e}")
    
    def _update_exchange_rate(self):
        """Update exchange rate (would call API in production)"""
        # In production, this would fetch from National Bank of Cambodia API
        # For now, use static rate
        pass
    
    def calculate_task_impact(self, 
                              task: Dict,
                              result: Dict,
                              grading_score: Optional[float] = None) -> EconomicMetrics:
        """
        Calculate economic impact for a single task
        
        Args:
            task: Task dictionary with metadata
            result: Evaluation result with execution time
            grading_score: Optional quality score (0-1)
            
        Returns:
            Economic metrics for the task
        """
        # Get occupation and wage
        occupation = task.get('occupation', 'general').lower().replace(' ', '_')
        monthly_wage_usd = self.wage_data.get(occupation, 400)  # Default wage
        hourly_wage_usd = monthly_wage_usd / self.WORKING_HOURS_PER_MONTH
        
        # Calculate time savings
        human_time_minutes = task['task_metadata'].get('estimated_time_minutes', 30)
        human_time_hours = human_time_minutes / 60
        
        ai_time_seconds = result.get('execution_time', 10)
        ai_time_hours = ai_time_seconds / 3600
        
        time_saved_hours = max(0, human_time_hours - ai_time_hours)
        
        # Calculate cost savings
        human_cost_usd = human_time_hours * hourly_wage_usd
        ai_cost_usd = self._calculate_ai_cost(ai_time_seconds, task.get('category', 'general'))
        
        cost_saved_usd = human_cost_usd - ai_cost_usd
        
        # Adjust for quality if grading score provided
        if grading_score is not None:
            # If AI quality is lower, reduce savings
            quality_adjustment = grading_score ** 2  # Quadratic penalty for low quality
            cost_saved_usd *= quality_adjustment
            time_saved_hours *= quality_adjustment
        
        cost_saved_khr = cost_saved_usd * self.exchange_rate
        
        # Calculate productivity gain
        productivity_gain = (time_saved_hours / human_time_hours * 100) if human_time_hours > 0 else 0
        
        # Calculate ROI
        implementation_cost = self._estimate_implementation_cost(task.get('category', 'general'))
        annual_savings = cost_saved_usd * self._estimate_task_frequency(task) * 12
        roi = (annual_savings / implementation_cost * 100) if implementation_cost > 0 else 0
        
        # Break-even calculation
        monthly_savings = cost_saved_usd * self._estimate_task_frequency(task)
        break_even_months = (implementation_cost / monthly_savings) if monthly_savings > 0 else float('inf')
        
        # GDP impact (basis points)
        sector = task.get('category', 'general')
        sector_gdp = self.SECTOR_GDP.get(sector, 1000) * 1_000_000  # Convert to USD
        gdp_impact_bps = (annual_savings / sector_gdp * 10000) if sector_gdp > 0 else 0
        
        # Jobs impact estimation
        jobs_affected = self._estimate_jobs_affected(sector, annual_savings)
        
        return EconomicMetrics(
            time_saved_hours=time_saved_hours,
            cost_saved_usd=cost_saved_usd,
            cost_saved_khr=cost_saved_khr,
            productivity_gain_percent=productivity_gain,
            roi_percentage=roi,
            break_even_months=break_even_months,
            annual_impact_usd=annual_savings,
            gdp_impact_basis_points=gdp_impact_bps,
            jobs_affected=jobs_affected,
            sector=sector
        )
    
    def _calculate_ai_cost(self, execution_time_seconds: float, category: str) -> float:
        """Calculate cost of AI execution"""
        # Simplified cost model
        # In reality, would consider API costs, compute costs, etc.
        
        base_cost_per_second = 0.0001  # USD
        
        # Category multipliers
        category_multipliers = {
            'finance': 1.5,  # More expensive due to compliance
            'healthcare': 1.5,
            'technology': 1.0,
            'agriculture': 0.8,
            'education': 0.8,
        }
        
        multiplier = category_multipliers.get(category, 1.0)
        
        return execution_time_seconds * base_cost_per_second * multiplier
    
    def _estimate_implementation_cost(self, category: str) -> float:
        """Estimate AI implementation cost by category"""
        # Simplified estimation
        category_costs = {
            'finance': 50000,
            'healthcare': 40000,
            'technology': 30000,
            'manufacturing': 45000,
            'agriculture': 15000,
            'education': 10000,
            'tourism': 20000,
        }
        
        return category_costs.get(category, 25000)
    
    def _estimate_task_frequency(self, task: Dict) -> int:
        """Estimate how often this task is performed per month"""
        # Based on task metadata and occupation
        
        difficulty = task['task_metadata'].get('difficulty_level', 3)
        
        # Higher difficulty = less frequent
        frequency_by_difficulty = {
            1: 100,  # Very simple, done many times daily
            2: 50,   # Simple, done daily
            3: 20,   # Medium, done weekly
            4: 5,    # Complex, done monthly
            5: 1,    # Very complex, done quarterly
        }
        
        return frequency_by_difficulty.get(difficulty, 10)
    
    def _estimate_jobs_affected(self, sector: str, annual_savings: float) -> int:
        """Estimate number of jobs affected by AI adoption"""
        # Rough estimation based on sector and savings
        
        avg_annual_wage = self.wage_data.get(f"{sector}_average", 400) * 12
        
        # Not all savings translate to job losses
        # Some create new opportunities
        impact_factor = 0.3  # 30% of savings might affect jobs
        
        jobs_affected = int((annual_savings * impact_factor) / avg_annual_wage)
        
        return max(0, jobs_affected)
    
    def analyze_batch_results(self, 
                              tasks: List[Dict],
                              results: List[Dict],
                              grades: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze economic impact for batch of results
        
        Args:
            tasks: List of task dictionaries
            results: List of evaluation results
            grades: Optional list of grading results
            
        Returns:
            Comprehensive economic analysis
        """
        # Create mapping for easy lookup
        task_map = {t['task_id']: t for t in tasks}
        grade_map = {}
        
        if grades:
            for grade in grades:
                grade_map[grade['task_id']] = grade.get('total_score', 1.0)
        
        # Calculate metrics for each result
        all_metrics = []
        
        for result in results:
            if not result.get('success'):
                continue
            
            task_id = result['task_id']
            task = task_map.get(task_id)
            
            if not task:
                continue
            
            grading_score = grade_map.get(task_id)
            
            metrics = self.calculate_task_impact(task, result, grading_score)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        return self._aggregate_metrics(all_metrics)
    
    def _aggregate_metrics(self, metrics_list: List[EconomicMetrics]) -> Dict:
        """Aggregate economic metrics across tasks"""
        
        if not metrics_list:
            return {"error": "No metrics to analyze"}
        
        # Overall aggregates
        total_time_saved = sum(m.time_saved_hours for m in metrics_list)
        total_cost_saved_usd = sum(m.cost_saved_usd for m in metrics_list)
        total_cost_saved_khr = sum(m.cost_saved_khr for m in metrics_list)
        total_annual_impact = sum(m.annual_impact_usd for m in metrics_list)
        
        # By sector
        sector_metrics = {}
        for metric in metrics_list:
            sector = metric.sector
            if sector not in sector_metrics:
                sector_metrics[sector] = {
                    'time_saved_hours': 0,
                    'cost_saved_usd': 0,
                    'annual_impact_usd': 0,
                    'productivity_gain_percent': [],
                    'roi_percentage': [],
                    'gdp_impact_bps': 0,
                    'jobs_affected': 0,
                    'task_count': 0
                }
            
            sector_metrics[sector]['time_saved_hours'] += metric.time_saved_hours
            sector_metrics[sector]['cost_saved_usd'] += metric.cost_saved_usd
            sector_metrics[sector]['annual_impact_usd'] += metric.annual_impact_usd
            sector_metrics[sector]['productivity_gain_percent'].append(metric.productivity_gain_percent)
            sector_metrics[sector]['roi_percentage'].append(metric.roi_percentage)
            sector_metrics[sector]['gdp_impact_bps'] += metric.gdp_impact_basis_points
            sector_metrics[sector]['jobs_affected'] += metric.jobs_affected
            sector_metrics[sector]['task_count'] += 1
        
        # Calculate sector averages
        for sector in sector_metrics:
            sector_metrics[sector]['avg_productivity_gain'] = np.mean(
                sector_metrics[sector]['productivity_gain_percent']
            )
            sector_metrics[sector]['avg_roi'] = np.mean(
                sector_metrics[sector]['roi_percentage']
            )
            # Remove list fields
            del sector_metrics[sector]['productivity_gain_percent']
            del sector_metrics[sector]['roi_percentage']
        
        # National impact estimation
        total_gdp_impact_bps = sum(m.gdp_impact_basis_points for m in metrics_list)
        total_jobs_affected = sum(m.jobs_affected for m in metrics_list)
        
        # Calculate adoption scenarios
        adoption_scenarios = self._calculate_adoption_scenarios(
            total_annual_impact,
            total_jobs_affected
        )
        
        return {
            'summary': {
                'total_tasks_analyzed': len(metrics_list),
                'total_time_saved_hours': round(total_time_saved, 2),
                'total_cost_saved_usd': round(total_cost_saved_usd, 2),
                'total_cost_saved_khr': round(total_cost_saved_khr, 0),
                'annual_impact_usd': round(total_annual_impact, 2),
                'annual_impact_khr': round(total_annual_impact * self.exchange_rate, 0),
                'avg_productivity_gain_percent': round(
                    np.mean([m.productivity_gain_percent for m in metrics_list]), 1
                ),
                'avg_roi_percentage': round(
                    np.mean([m.roi_percentage for m in metrics_list]), 1
                ),
                'avg_break_even_months': round(
                    np.mean([m.break_even_months for m in metrics_list 
                           if m.break_even_months != float('inf')]), 1
                ),
                'gdp_impact_basis_points': round(total_gdp_impact_bps, 2),
                'estimated_jobs_affected': total_jobs_affected
            },
            'by_sector': sector_metrics,
            'adoption_scenarios': adoption_scenarios,
            'exchange_rate_used': self.exchange_rate,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_adoption_scenarios(self, 
                                     base_annual_impact: float,
                                     base_jobs_affected: int) -> Dict:
        """Calculate different AI adoption scenarios"""
        
        scenarios = {
            'conservative': {
                'adoption_rate': 0.1,  # 10% adoption
                'timeline_years': 5,
                'annual_impact_usd': base_annual_impact * 0.1,
                'cumulative_impact_usd': base_annual_impact * 0.1 * 5,
                'jobs_affected': int(base_jobs_affected * 0.1),
                'new_jobs_created': int(base_jobs_affected * 0.1 * 0.3),  # 30% new job creation
            },
            'moderate': {
                'adoption_rate': 0.3,  # 30% adoption
                'timeline_years': 3,
                'annual_impact_usd': base_annual_impact * 0.3,
                'cumulative_impact_usd': base_annual_impact * 0.3 * 3,
                'jobs_affected': int(base_jobs_affected * 0.3),
                'new_jobs_created': int(base_jobs_affected * 0.3 * 0.4),
            },
            'aggressive': {
                'adoption_rate': 0.6,  # 60% adoption
                'timeline_years': 2,
                'annual_impact_usd': base_annual_impact * 0.6,
                'cumulative_impact_usd': base_annual_impact * 0.6 * 2,
                'jobs_affected': int(base_jobs_affected * 0.6),
                'new_jobs_created': int(base_jobs_affected * 0.6 * 0.5),
            }
        }
        
        # Calculate net job impact
        for scenario in scenarios.values():
            scenario['net_job_impact'] = scenario['jobs_affected'] - scenario['new_jobs_created']
            scenario['economic_growth_potential'] = round(
                scenario['cumulative_impact_usd'] / 1_000_000, 2  # In millions
            )
        
        return scenarios
    
    def generate_policy_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """
        Generate policy recommendations based on economic analysis
        
        Args:
            analysis_results: Results from analyze_batch_results
            
        Returns:
            List of policy recommendations
        """
        recommendations = []
        
        # Check overall ROI
        avg_roi = analysis_results['summary'].get('avg_roi_percentage', 0)
        
        if avg_roi > 200:
            recommendations.append({
                'priority': 'high',
                'category': 'investment',
                'recommendation': 'Establish national AI adoption fund',
                'rationale': f'Average ROI of {avg_roi:.0f}% indicates strong economic potential',
                'estimated_impact': 'Could accelerate adoption by 2-3 years'
            })
        
        # Check jobs impact
        jobs_affected = analysis_results['summary'].get('estimated_jobs_affected', 0)
        
        if jobs_affected > 100:
            recommendations.append({
                'priority': 'high',
                'category': 'workforce',
                'recommendation': 'Implement AI reskilling programs',
                'rationale': f'Estimated {jobs_affected} jobs will be affected',
                'estimated_impact': 'Reduce unemployment risk by 60%'
            })
        
        # Sector-specific recommendations
        for sector, metrics in analysis_results['by_sector'].items():
            if metrics['avg_productivity_gain'] > 50:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'sector_specific',
                    'recommendation': f'Create {sector} AI innovation hub',
                    'rationale': f'{sector} shows {metrics["avg_productivity_gain"]:.0f}% productivity potential',
                    'estimated_impact': f'Could generate ${metrics["annual_impact_usd"]:,.0f} annually'
                })
        
        # Infrastructure recommendations
        if analysis_results['summary']['total_tasks_analyzed'] > 50:
            recommendations.append({
                'priority': 'medium',
                'category': 'infrastructure',
                'recommendation': 'Upgrade digital infrastructure for AI workloads',
                'rationale': 'Current analysis shows significant computational needs',
                'estimated_impact': 'Reduce AI processing costs by 30%'
            })
        
        # Education recommendations
        recommendations.append({
            'priority': 'high',
            'category': 'education',
            'recommendation': 'Integrate AI literacy in national curriculum',
            'rationale': 'Prepare workforce for AI-augmented economy',
            'estimated_impact': 'Increase AI-ready workforce by 10x in 5 years'
        })
        
        return sorted(recommendations, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
    
    def export_report(self, 
                      analysis_results: Dict,
                      output_path: str,
                      format: str = 'json'):
        """
        Export economic analysis report
        
        Args:
            analysis_results: Results from analyze_batch_results
            output_path: Path to save report
            format: Output format (json, csv, html)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            # Convert to DataFrame for CSV export
            summary_df = pd.DataFrame([analysis_results['summary']])
            summary_df.to_csv(output_path.with_suffix('.summary.csv'), index=False)
            
            if 'by_sector' in analysis_results:
                sector_df = pd.DataFrame(analysis_results['by_sector']).T
                sector_df.to_csv(output_path.with_suffix('.sectors.csv'))
        
        elif format == 'html':
            # Generate HTML report
            html_content = self._generate_html_report(analysis_results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        logger.info(f"Report exported to {output_path}")
    
    def _generate_html_report(self, analysis_results: Dict) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>GDPval Economic Impact Report - Cambodia</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #003366; }}
                h2 {{ color: #006699; border-bottom: 2px solid #006699; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #009900; }}
                .warning {{ color: #ff6600; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>🇰🇭 Economic Impact Analysis - AI Adoption in Cambodia</h1>
            <p>Generated: {analysis_results.get('analysis_timestamp', 'N/A')}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Tasks Analyzed</td>
                        <td>{analysis_results['summary']['total_tasks_analyzed']}</td>
                    </tr>
                    <tr>
                        <td>Total Cost Saved</td>
                        <td class="metric">${analysis_results['summary']['total_cost_saved_usd']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Total Cost Saved (KHR)</td>
                        <td class="metric">៛{analysis_results['summary']['total_cost_saved_khr']:,.0f}</td>
                    </tr>
                    <tr>
                        <td>Annual Economic Impact</td>
                        <td class="metric">${analysis_results['summary']['annual_impact_usd']:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Average ROI</td>
                        <td class="metric">{analysis_results['summary']['avg_roi_percentage']:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Average Productivity Gain</td>
                        <td class="metric">{analysis_results['summary']['avg_productivity_gain_percent']:.1f}%</td>
                    </tr>
                    <tr>
                        <td>GDP Impact</td>
                        <td>{analysis_results['summary']['gdp_impact_basis_points']:.2f} basis points</td>
                    </tr>
                    <tr>
                        <td>Jobs Potentially Affected</td>
                        <td class="warning">{analysis_results['summary']['estimated_jobs_affected']:,}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Sector Analysis</h2>
                <table>
                    <tr>
                        <th>Sector</th>
                        <th>Tasks</th>
                        <th>Time Saved (hrs)</th>
                        <th>Cost Saved (USD)</th>
                        <th>Annual Impact (USD)</th>
                        <th>Avg ROI (%)</th>
                    </tr>
        """
        
        for sector, metrics in analysis_results.get('by_sector', {}).items():
            html += f"""
                    <tr>
                        <td>{sector.title()}</td>
                        <td>{metrics['task_count']}</td>
                        <td>{metrics['time_saved_hours']:.1f}</td>
                        <td>${metrics['cost_saved_usd']:,.2f}</td>
                        <td>${metrics['annual_impact_usd']:,.2f}</td>
                        <td>{metrics.get('avg_roi', 0):.1f}%</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Adoption Scenarios</h2>
        """
        
        for scenario_name, scenario in analysis_results.get('adoption_scenarios', {}).items():
            html += f"""
                <h3>{scenario_name.title()} Scenario</h3>
                <ul>
                    <li>Adoption Rate: {scenario['adoption_rate']*100:.0f}%</li>
                    <li>Timeline: {scenario['timeline_years']} years</li>
                    <li>Annual Impact: ${scenario['annual_impact_usd']:,.0f}</li>
                    <li>Cumulative Impact: ${scenario['cumulative_impact_usd']:,.0f}</li>
                    <li>Net Job Impact: {scenario['net_job_impact']:,}</li>
                </ul>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CambodianEconomicAnalyzer()
    
    # Sample task
    sample_task = {
        'task_id': 'FIN-KH-001',
        'occupation': 'Financial Analyst',
        'category': 'finance',
        'task_metadata': {
            'estimated_time_minutes': 45,
            'difficulty_level': 3,
            'wage_per_hour_usd': 25  # Optional, will use default if not provided
        }
    }
    
    # Sample result
    sample_result = {
        'task_id': 'FIN-KH-001',
        'execution_time': 30,  # seconds
        'success': True
    }
    
    # Calculate impact for single task
    metrics = analyzer.calculate_task_impact(sample_task, sample_result, grading_score=0.85)
    
    print("=" * 60)
    print("Economic Impact Analysis - Single Task")
    print("=" * 60)
    print(f"Time Saved: {metrics.time_saved_hours:.2f} hours")
    print(f"Cost Saved: ${metrics.cost_saved_usd:.2f} (៛{metrics.cost_saved_khr:.0f})")
    print(f"Productivity Gain: {metrics.productivity_gain_percent:.1f}%")
    print(f"ROI: {metrics.roi_percentage:.1f}%")
    print(f"Break-even: {metrics.break_even_months:.1f} months")
    print(f"Annual Impact: ${metrics.annual_impact_usd:.2f}")
    print(f"GDP Impact: {metrics.gdp_impact_basis_points:.3f} basis points")
    
    # Batch analysis
    tasks = [sample_task] * 10  # Simulate 10 tasks
    results = [sample_result] * 10
    
    batch_analysis = analyzer.analyze_batch_results(tasks, results)
    
    print("\n" + "=" * 60)
    print("Economic Impact Analysis - Batch")
    print("=" * 60)
    print(json.dumps(batch_analysis['summary'], indent=2))
    
    # Generate policy recommendations
    recommendations = analyzer.generate_policy_recommendations(batch_analysis)
    
    print("\n" + "=" * 60)
    print("Policy Recommendations")
    print("=" * 60)
    for rec in recommendations:
        print(f"\n[{rec['priority'].upper()}] {rec['category'].title()}")
        print(f"  Recommendation: {rec['recommendation']}")
        print(f"  Rationale: {rec['rationale']}")
        print(f"  Impact: {rec['estimated_impact']}")
    
    # Export report
    # analyzer.export_report(batch_analysis, "economic_report.html", format="html")