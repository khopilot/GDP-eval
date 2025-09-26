"""
Economic Impact Calculator
Calculates comprehensive economic impact of AI adoption
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImpactMetrics:
    """Container for impact metrics"""
    productivity_gain_hours: float
    cost_saved_usd: float
    revenue_increase_usd: float
    quality_improvement_percent: float
    customer_satisfaction_increase: float
    time_to_market_reduction_days: float
    error_reduction_percent: float
    compliance_improvement_percent: float


class ImpactCalculator:
    """
    Calculates economic and productivity impact of AI implementations
    """

    # Cambodia labor statistics
    LABOR_STATISTICS = {
        'avg_work_hours_per_day': 8,
        'work_days_per_year': 260,
        'avg_hourly_wage_usd': 2.5,
        'productivity_baseline': 0.6,  # 60% productivity baseline
    }

    # Impact multipliers by task type
    TASK_MULTIPLIERS = {
        'data_entry': 3.5,
        'document_processing': 2.8,
        'customer_service': 2.2,
        'analysis': 1.8,
        'decision_support': 1.5,
        'quality_control': 2.5,
        'translation': 3.0,
        'content_generation': 2.0,
    }

    def calculate_productivity_impact(
        self,
        task_type: str,
        hours_per_task: float,
        tasks_per_day: int,
        automation_level: float,
        workers_affected: int
    ) -> Dict[str, float]:
        """
        Calculate productivity impact of AI implementation

        Args:
            task_type: Type of task being automated
            hours_per_task: Hours required per task
            tasks_per_day: Number of tasks per day
            automation_level: Level of automation (0-1)
            workers_affected: Number of workers affected

        Returns:
            Productivity impact metrics
        """
        # Get task multiplier
        multiplier = self.TASK_MULTIPLIERS.get(task_type, 1.0)

        # Current time spent
        current_time_per_day = hours_per_task * tasks_per_day * workers_affected

        # Time saved through automation
        time_saved_per_day = current_time_per_day * automation_level * (multiplier / 3.5)

        # Annual time saved
        annual_time_saved = time_saved_per_day * self.LABOR_STATISTICS['work_days_per_year']

        # Productivity gain
        productivity_gain = (time_saved_per_day / current_time_per_day) * 100 if current_time_per_day > 0 else 0

        # New tasks possible with saved time
        additional_tasks = (time_saved_per_day / hours_per_task) if hours_per_task > 0 else 0

        # Cost savings
        cost_saved_per_day = time_saved_per_day * self.LABOR_STATISTICS['avg_hourly_wage_usd']
        annual_cost_saved = cost_saved_per_day * self.LABOR_STATISTICS['work_days_per_year']

        return {
            'time_saved_per_day_hours': time_saved_per_day,
            'annual_time_saved_hours': annual_time_saved,
            'productivity_gain_percent': productivity_gain,
            'additional_tasks_per_day': additional_tasks,
            'cost_saved_per_day_usd': cost_saved_per_day,
            'annual_cost_saved_usd': annual_cost_saved,
            'efficiency_multiplier': multiplier,
            'workers_hours_freed': time_saved_per_day
        }

    def calculate_quality_impact(
        self,
        current_error_rate: float,
        ai_error_rate: float,
        errors_per_day: int,
        cost_per_error_usd: float
    ) -> Dict[str, float]:
        """
        Calculate quality improvement impact

        Args:
            current_error_rate: Current error rate (0-1)
            ai_error_rate: AI system error rate (0-1)
            errors_per_day: Number of errors per day
            cost_per_error_usd: Cost of each error

        Returns:
            Quality impact metrics
        """
        # Error reduction
        error_reduction_rate = max(0, current_error_rate - ai_error_rate)
        error_reduction_percent = (error_reduction_rate / current_error_rate * 100) if current_error_rate > 0 else 0

        # Errors prevented
        errors_prevented_per_day = errors_per_day * error_reduction_rate
        annual_errors_prevented = errors_prevented_per_day * self.LABOR_STATISTICS['work_days_per_year']

        # Cost savings from error prevention
        daily_savings = errors_prevented_per_day * cost_per_error_usd
        annual_savings = annual_errors_prevented * cost_per_error_usd

        # Quality score improvement (simplified)
        quality_improvement = (1 - ai_error_rate) / (1 - current_error_rate) if current_error_rate < 1 else float('inf')

        return {
            'error_reduction_percent': error_reduction_percent,
            'errors_prevented_per_day': errors_prevented_per_day,
            'annual_errors_prevented': annual_errors_prevented,
            'daily_quality_savings_usd': daily_savings,
            'annual_quality_savings_usd': annual_savings,
            'quality_improvement_factor': quality_improvement,
            'new_error_rate': ai_error_rate,
            'accuracy_improvement': (1 - ai_error_rate) - (1 - current_error_rate)
        }

    def calculate_customer_impact(
        self,
        current_response_time_hours: float,
        ai_response_time_hours: float,
        customers_per_day: int,
        customer_value_usd: float,
        current_satisfaction: float
    ) -> Dict[str, float]:
        """
        Calculate customer experience impact

        Args:
            current_response_time_hours: Current response time
            ai_response_time_hours: AI-enabled response time
            customers_per_day: Number of customers per day
            customer_value_usd: Average customer value
            current_satisfaction: Current satisfaction score (0-100)

        Returns:
            Customer impact metrics
        """
        # Response time improvement
        time_reduction = current_response_time_hours - ai_response_time_hours
        time_reduction_percent = (time_reduction / current_response_time_hours * 100) if current_response_time_hours > 0 else 0

        # Customer satisfaction improvement (empirical formula)
        satisfaction_boost = min(20, time_reduction_percent * 0.4)
        new_satisfaction = min(100, current_satisfaction + satisfaction_boost)

        # Customer retention impact (1% satisfaction = 0.5% retention)
        retention_improvement = satisfaction_boost * 0.005

        # Revenue impact
        daily_customer_value = customers_per_day * customer_value_usd
        retention_revenue_increase = daily_customer_value * retention_improvement
        annual_revenue_increase = retention_revenue_increase * 365

        # Service capacity increase
        capacity_increase = (current_response_time_hours / ai_response_time_hours) if ai_response_time_hours > 0 else float('inf')
        additional_customers = customers_per_day * (capacity_increase - 1)

        return {
            'response_time_reduction_hours': time_reduction,
            'response_time_reduction_percent': time_reduction_percent,
            'satisfaction_increase': satisfaction_boost,
            'new_satisfaction_score': new_satisfaction,
            'retention_improvement_percent': retention_improvement * 100,
            'daily_revenue_increase_usd': retention_revenue_increase,
            'annual_revenue_increase_usd': annual_revenue_increase,
            'capacity_increase_factor': capacity_increase,
            'additional_customers_per_day': additional_customers
        }

    def calculate_compliance_impact(
        self,
        current_compliance_rate: float,
        ai_compliance_rate: float,
        compliance_checks_per_day: int,
        penalty_per_violation_usd: float
    ) -> Dict[str, float]:
        """
        Calculate regulatory compliance impact

        Args:
            current_compliance_rate: Current compliance rate (0-1)
            ai_compliance_rate: AI-enabled compliance rate
            compliance_checks_per_day: Number of compliance checks
            penalty_per_violation_usd: Penalty per violation

        Returns:
            Compliance impact metrics
        """
        # Compliance improvement
        compliance_improvement = ai_compliance_rate - current_compliance_rate
        compliance_improvement_percent = (compliance_improvement / current_compliance_rate * 100) if current_compliance_rate > 0 else 0

        # Violations prevented
        current_violations = compliance_checks_per_day * (1 - current_compliance_rate)
        ai_violations = compliance_checks_per_day * (1 - ai_compliance_rate)
        violations_prevented = current_violations - ai_violations

        # Penalty savings
        daily_penalty_savings = violations_prevented * penalty_per_violation_usd
        annual_penalty_savings = daily_penalty_savings * self.LABOR_STATISTICS['work_days_per_year']

        # Risk reduction
        risk_reduction = (1 - ai_compliance_rate) / (1 - current_compliance_rate) if current_compliance_rate < 1 else 0

        return {
            'compliance_improvement_percent': compliance_improvement_percent,
            'new_compliance_rate': ai_compliance_rate * 100,
            'violations_prevented_per_day': violations_prevented,
            'daily_penalty_savings_usd': daily_penalty_savings,
            'annual_penalty_savings_usd': annual_penalty_savings,
            'risk_reduction_factor': 1 - risk_reduction,
            'compliance_confidence': ai_compliance_rate
        }

    def calculate_innovation_impact(
        self,
        time_to_market_days: float,
        ai_time_reduction_percent: float,
        project_value_usd: float,
        projects_per_year: int
    ) -> Dict[str, float]:
        """
        Calculate innovation and time-to-market impact

        Args:
            time_to_market_days: Current time to market
            ai_time_reduction_percent: AI-enabled reduction
            project_value_usd: Value per project
            projects_per_year: Number of projects

        Returns:
            Innovation impact metrics
        """
        # Time to market reduction
        time_saved_days = time_to_market_days * (ai_time_reduction_percent / 100)
        new_time_to_market = time_to_market_days - time_saved_days

        # Additional projects possible
        time_efficiency = time_to_market_days / new_time_to_market if new_time_to_market > 0 else 1
        additional_projects = projects_per_year * (time_efficiency - 1)

        # Revenue impact from faster delivery
        daily_project_value = project_value_usd / time_to_market_days if time_to_market_days > 0 else 0
        early_revenue = daily_project_value * time_saved_days * projects_per_year

        # Innovation capacity
        innovation_capacity = time_efficiency
        resource_efficiency = 1 + (ai_time_reduction_percent / 100)

        return {
            'time_to_market_reduction_days': time_saved_days,
            'new_time_to_market_days': new_time_to_market,
            'additional_projects_per_year': additional_projects,
            'early_revenue_capture_usd': early_revenue,
            'innovation_capacity_increase': (innovation_capacity - 1) * 100,
            'resource_efficiency_gain': (resource_efficiency - 1) * 100,
            'competitive_advantage_days': time_saved_days
        }

    def calculate_total_impact(
        self,
        productivity_impact: Dict,
        quality_impact: Dict,
        customer_impact: Dict,
        compliance_impact: Optional[Dict] = None,
        innovation_impact: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Calculate total economic impact

        Args:
            productivity_impact: Productivity metrics
            quality_impact: Quality metrics
            customer_impact: Customer metrics
            compliance_impact: Compliance metrics (optional)
            innovation_impact: Innovation metrics (optional)

        Returns:
            Total impact summary
        """
        # Aggregate financial impact
        total_cost_savings = (
            productivity_impact.get('annual_cost_saved_usd', 0) +
            quality_impact.get('annual_quality_savings_usd', 0) +
            (compliance_impact.get('annual_penalty_savings_usd', 0) if compliance_impact else 0)
        )

        total_revenue_increase = (
            customer_impact.get('annual_revenue_increase_usd', 0) +
            (innovation_impact.get('early_revenue_capture_usd', 0) if innovation_impact else 0)
        )

        total_impact_usd = total_cost_savings + total_revenue_increase

        # Aggregate operational improvements
        total_time_saved = productivity_impact.get('annual_time_saved_hours', 0)
        total_errors_prevented = quality_impact.get('annual_errors_prevented', 0)
        total_customers_gained = customer_impact.get('additional_customers_per_day', 0) * 365

        # Calculate ROI (simplified)
        # Assuming implementation cost is 20% of annual impact
        estimated_implementation_cost = total_impact_usd * 0.2
        roi = ((total_impact_usd - estimated_implementation_cost) / estimated_implementation_cost * 100) if estimated_implementation_cost > 0 else 0

        summary = {
            'financial_impact': {
                'total_annual_impact_usd': total_impact_usd,
                'total_cost_savings_usd': total_cost_savings,
                'total_revenue_increase_usd': total_revenue_increase,
                'estimated_roi_percent': roi,
                'payback_period_months': (estimated_implementation_cost / (total_impact_usd / 12)) if total_impact_usd > 0 else float('inf')
            },
            'operational_impact': {
                'total_time_saved_hours': total_time_saved,
                'total_errors_prevented': total_errors_prevented,
                'total_customers_gained': total_customers_gained,
                'productivity_gain_percent': productivity_impact.get('productivity_gain_percent', 0),
                'quality_improvement_percent': quality_impact.get('error_reduction_percent', 0)
            },
            'strategic_impact': {
                'customer_satisfaction_increase': customer_impact.get('satisfaction_increase', 0),
                'compliance_improvement': compliance_impact.get('compliance_improvement_percent', 0) if compliance_impact else 0,
                'innovation_capacity_increase': innovation_impact.get('innovation_capacity_increase', 0) if innovation_impact else 0,
                'competitive_advantage': 'High' if roi > 100 else 'Medium' if roi > 50 else 'Low'
            },
            'implementation_metrics': {
                'estimated_cost_usd': estimated_implementation_cost,
                'time_to_value_months': 3 if roi > 100 else 6 if roi > 50 else 12,
                'risk_level': 'Low' if roi > 100 else 'Medium' if roi > 50 else 'High',
                'recommendation': 'Strongly Recommended' if roi > 100 else 'Recommended' if roi > 50 else 'Consider Carefully'
            }
        }

        return summary

    def export_impact_report(
        self,
        impact_summary: Dict,
        filepath: str
    ):
        """Export impact analysis to JSON"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'impact_analysis': impact_summary,
            'methodology': 'GDP-eval Impact Calculator',
            'currency': 'USD'
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Impact report exported to {filepath}")