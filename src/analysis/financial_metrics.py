"""
Financial Metrics Calculator
Calculates financial KPIs and metrics for AI investment decisions
"""

import numpy as np
try:
    import numpy_financial as npf
except ImportError:
    npf = None
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """Container for financial metrics"""
    roi: float
    npv: float
    irr: float
    payback_period_months: float
    break_even_point: float
    total_cost_ownership: float
    cost_per_transaction: float
    revenue_per_user: float
    profit_margin: float
    ebitda: float


class FinancialCalculator:
    """
    Calculates comprehensive financial metrics for AI implementations
    """

    # Cambodia-specific financial parameters
    DISCOUNT_RATE = 0.10  # 10% discount rate
    INFLATION_RATE = 0.03  # 3% inflation
    CORPORATE_TAX_RATE = 0.20  # 20% corporate tax
    USD_TO_KHR = 4100  # Exchange rate

    # Operational cost factors
    OPERATIONAL_COST_FACTORS = {
        'electricity_per_kwh_usd': 0.15,
        'internet_per_month_usd': 100,
        'office_rent_per_sqm_usd': 15,
        'avg_salary_it_usd': 800,
        'avg_salary_operator_usd': 300,
        'training_cost_per_person_usd': 200,
    }

    def calculate_roi(
        self,
        initial_investment: float,
        annual_returns: List[float],
        operational_costs: List[float]
    ) -> float:
        """
        Calculate Return on Investment

        Args:
            initial_investment: Initial investment in USD
            annual_returns: List of annual returns
            operational_costs: List of annual operational costs

        Returns:
            ROI percentage
        """
        total_returns = sum(annual_returns)
        total_costs = initial_investment + sum(operational_costs)

        if total_costs == 0:
            return float('inf')

        roi = ((total_returns - total_costs) / total_costs) * 100
        return roi

    def calculate_npv(
        self,
        initial_investment: float,
        cash_flows: List[float],
        discount_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Net Present Value

        Args:
            initial_investment: Initial investment (negative value)
            cash_flows: List of future cash flows
            discount_rate: Discount rate (default: class rate)

        Returns:
            NPV in USD
        """
        if discount_rate is None:
            discount_rate = self.DISCOUNT_RATE

        npv = -initial_investment
        for i, cash_flow in enumerate(cash_flows, 1):
            npv += cash_flow / ((1 + discount_rate) ** i)

        return npv

    def calculate_irr(
        self,
        initial_investment: float,
        cash_flows: List[float]
    ) -> float:
        """
        Calculate Internal Rate of Return

        Args:
            initial_investment: Initial investment
            cash_flows: List of cash flows

        Returns:
            IRR as decimal
        """
        # Combine initial investment with cash flows
        all_cash_flows = [-initial_investment] + cash_flows

        try:
            if npf is not None:
                # Use numpy_financial if available
                irr = npf.irr(all_cash_flows)
            else:
                # Fallback to custom IRR calculation using Newton's method
                # IRR is the discount rate where NPV = 0
                def npv_func(rate):
                    return sum([cf / (1 + rate) ** i for i, cf in enumerate(all_cash_flows)])

                # Newton's method to find root
                rate = 0.1  # Initial guess
                for _ in range(100):  # Max iterations
                    npv = npv_func(rate)
                    if abs(npv) < 0.01:  # Close enough to zero
                        break
                    # Derivative approximation
                    dnpv = (npv_func(rate + 0.0001) - npv) / 0.0001
                    if abs(dnpv) < 0.0001:  # Avoid division by zero
                        break
                    rate = rate - npv / dnpv
                irr = rate
            return irr if not np.isnan(irr) else 0.0
        except Exception as e:
            logger.warning(f"IRR calculation failed: {e}")
            return 0.0

    def calculate_payback_period(
        self,
        initial_investment: float,
        monthly_returns: List[float],
        monthly_costs: List[float]
    ) -> float:
        """
        Calculate payback period in months

        Args:
            initial_investment: Initial investment
            monthly_returns: Monthly returns
            monthly_costs: Monthly operational costs

        Returns:
            Payback period in months
        """
        cumulative_cash_flow = 0
        for month, (return_amt, cost) in enumerate(zip(monthly_returns, monthly_costs), 1):
            net_cash_flow = return_amt - cost
            cumulative_cash_flow += net_cash_flow

            if cumulative_cash_flow >= initial_investment:
                # Interpolate for partial month
                prev_cumulative = cumulative_cash_flow - net_cash_flow
                partial_month = (initial_investment - prev_cumulative) / net_cash_flow
                return month - 1 + partial_month

        # If payback period exceeds available data
        return float('inf')

    def calculate_tco(
        self,
        hardware_cost: float,
        software_cost: float,
        implementation_cost: float,
        training_cost: float,
        operational_years: int = 3,
        maintenance_rate: float = 0.15
    ) -> float:
        """
        Calculate Total Cost of Ownership

        Args:
            hardware_cost: Hardware costs
            software_cost: Software/licensing costs
            implementation_cost: Implementation costs
            training_cost: Training costs
            operational_years: Years of operation
            maintenance_rate: Annual maintenance as % of initial cost

        Returns:
            Total cost of ownership
        """
        initial_cost = hardware_cost + software_cost + implementation_cost + training_cost

        # Annual operational costs
        annual_maintenance = initial_cost * maintenance_rate
        annual_operational = (
            self.OPERATIONAL_COST_FACTORS['electricity_per_kwh_usd'] * 1000 * 12 +
            self.OPERATIONAL_COST_FACTORS['internet_per_month_usd'] * 12
        )

        # Total operational cost
        operational_cost = (annual_maintenance + annual_operational) * operational_years

        # Account for inflation
        inflation_adjusted = operational_cost * (1 + self.INFLATION_RATE) ** (operational_years / 2)

        tco = initial_cost + inflation_adjusted
        return tco

    def calculate_unit_economics(
        self,
        total_cost: float,
        transaction_volume: int,
        user_count: int,
        revenue: float
    ) -> Dict[str, float]:
        """
        Calculate unit economics metrics

        Args:
            total_cost: Total operational cost
            transaction_volume: Number of transactions
            user_count: Number of users
            revenue: Total revenue

        Returns:
            Unit economics metrics
        """
        cost_per_transaction = total_cost / transaction_volume if transaction_volume > 0 else 0
        cost_per_user = total_cost / user_count if user_count > 0 else 0
        revenue_per_transaction = revenue / transaction_volume if transaction_volume > 0 else 0
        revenue_per_user = revenue / user_count if user_count > 0 else 0

        return {
            'cost_per_transaction': cost_per_transaction,
            'cost_per_user': cost_per_user,
            'revenue_per_transaction': revenue_per_transaction,
            'revenue_per_user': revenue_per_user,
            'unit_profit': revenue_per_transaction - cost_per_transaction,
            'ltv_cac_ratio': revenue_per_user / cost_per_user if cost_per_user > 0 else 0
        }

    def calculate_break_even(
        self,
        fixed_costs: float,
        variable_cost_per_unit: float,
        price_per_unit: float
    ) -> Dict[str, float]:
        """
        Calculate break-even point

        Args:
            fixed_costs: Fixed costs
            variable_cost_per_unit: Variable cost per unit
            price_per_unit: Selling price per unit

        Returns:
            Break-even metrics
        """
        if price_per_unit <= variable_cost_per_unit:
            return {
                'break_even_units': float('inf'),
                'break_even_revenue': float('inf'),
                'contribution_margin': 0,
                'margin_of_safety': 0
            }

        contribution_margin = price_per_unit - variable_cost_per_unit
        break_even_units = fixed_costs / contribution_margin
        break_even_revenue = break_even_units * price_per_unit

        return {
            'break_even_units': break_even_units,
            'break_even_revenue': break_even_revenue,
            'contribution_margin': contribution_margin,
            'contribution_margin_ratio': contribution_margin / price_per_unit
        }

    def calculate_financial_ratios(
        self,
        revenue: float,
        costs: float,
        assets: float,
        liabilities: float,
        equity: float
    ) -> Dict[str, float]:
        """
        Calculate key financial ratios

        Args:
            revenue: Total revenue
            costs: Total costs
            assets: Total assets
            liabilities: Total liabilities
            equity: Total equity

        Returns:
            Financial ratios
        """
        profit = revenue - costs
        ebitda = profit + (costs * 0.1)  # Simplified EBITDA calculation

        ratios = {
            'profit_margin': (profit / revenue * 100) if revenue > 0 else 0,
            'return_on_assets': (profit / assets * 100) if assets > 0 else 0,
            'return_on_equity': (profit / equity * 100) if equity > 0 else 0,
            'debt_to_equity': (liabilities / equity) if equity > 0 else 0,
            'current_ratio': assets / liabilities if liabilities > 0 else float('inf'),
            'ebitda_margin': (ebitda / revenue * 100) if revenue > 0 else 0
        }

        return ratios

    def sensitivity_analysis(
        self,
        base_npv: float,
        initial_investment: float,
        cash_flows: List[float],
        variables: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on NPV

        Args:
            base_npv: Base case NPV
            initial_investment: Initial investment
            cash_flows: Cash flows
            variables: Variables to test with % changes

        Returns:
            Sensitivity analysis results
        """
        results = {}

        for variable, change_range in variables.items():
            sensitivities = {}

            for change in [-0.2, -0.1, 0, 0.1, 0.2]:  # ±20% changes
                if variable == 'discount_rate':
                    new_rate = self.DISCOUNT_RATE * (1 + change)
                    new_npv = self.calculate_npv(initial_investment, cash_flows, new_rate)
                elif variable == 'cash_flows':
                    adjusted_flows = [cf * (1 + change) for cf in cash_flows]
                    new_npv = self.calculate_npv(initial_investment, adjusted_flows)
                elif variable == 'investment':
                    new_investment = initial_investment * (1 + change)
                    new_npv = self.calculate_npv(new_investment, cash_flows)
                else:
                    new_npv = base_npv

                sensitivities[f"{change:+.0%}"] = {
                    'npv': new_npv,
                    'change_from_base': new_npv - base_npv,
                    'percent_change': ((new_npv - base_npv) / abs(base_npv) * 100) if base_npv != 0 else 0
                }

            results[variable] = sensitivities

        return results

    def convert_currency(
        self,
        amount: float,
        from_currency: str = 'USD',
        to_currency: str = 'KHR'
    ) -> float:
        """
        Convert between USD and KHR

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Converted amount
        """
        if from_currency == to_currency:
            return amount

        if from_currency == 'USD' and to_currency == 'KHR':
            return amount * self.USD_TO_KHR
        elif from_currency == 'KHR' and to_currency == 'USD':
            return amount / self.USD_TO_KHR
        else:
            raise ValueError(f"Unsupported currency conversion: {from_currency} to {to_currency}")

    def generate_financial_report(
        self,
        metrics: FinancialMetrics,
        currency: str = 'USD'
    ) -> str:
        """
        Generate formatted financial report

        Args:
            metrics: Financial metrics
            currency: Currency for display

        Returns:
            Formatted report string
        """
        report = f"""
Financial Analysis Report
========================
Currency: {currency}
Date: {datetime.now().strftime('%Y-%m-%d')}

Key Metrics:
-----------
ROI: {metrics.roi:.2f}%
NPV: {currency} {metrics.npv:,.2f}
IRR: {metrics.irr:.2%}
Payback Period: {metrics.payback_period_months:.1f} months
Break-Even: {metrics.break_even_point:,.0f} units

Operational Metrics:
-------------------
Total Cost of Ownership: {currency} {metrics.total_cost_ownership:,.2f}
Cost per Transaction: {currency} {metrics.cost_per_transaction:.2f}
Revenue per User: {currency} {metrics.revenue_per_user:.2f}

Profitability:
-------------
Profit Margin: {metrics.profit_margin:.2f}%
EBITDA: {currency} {metrics.ebitda:,.2f}
"""
        return report