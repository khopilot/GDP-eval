#!/usr/bin/env python3
"""
Multi-Dimensional Evaluation Test Script
Enterprise-grade comprehensive AI model evaluation

Author: Nicolas Delrieu, AI Consultant
Phone: +855 92 332 554
"""

import asyncio
import os
import sys
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.providers.grok_provider import GrokProvider
from src.evaluation.multidimensional import (
    EvaluationOrchestrator,
    EvaluationConfiguration,
    EvaluationSuite,
    ReportGenerator
)


class MultidimensionalEvaluationRunner:
    """
    Comprehensive evaluation runner for testing all evaluation modules
    """

    def __init__(self):
        self.api_key = self._get_api_key()
        self.model_provider = None
        self.orchestrator = None
        self.report_generator = None

    def _get_api_key(self) -> str:
        """Get API key from environment or return test key"""
        api_key = os.getenv('GROK_API_KEY', 'YOUR-API-KEY-HERE')
        if not api_key or api_key == 'your_grok_api_key_here':
            raise ValueError(
                "Please set GROK_API_KEY environment variable or update the test script with your API key"
            )
        return api_key

    def initialize_providers(self):
        """Initialize model provider and evaluation components"""
        print("🚀 Initializing Grok provider...")
        self.model_provider = GrokProvider(
            api_key=self.api_key,
            model="grok-3",
            timeout=180  # 3 minutes for comprehensive evaluation
        )

        print("🧠 Initializing evaluation orchestrator...")
        config = EvaluationConfiguration(
            suite_type=EvaluationSuite.COMPREHENSIVE,
            modules=["capability", "safety", "robustness", "consistency", "behavioral"],
            parallel_execution=True,
            timeout_seconds=3600,  # 1 hour
            save_intermediate=True,
            output_format="json"
        )
        self.orchestrator = EvaluationOrchestrator(config)

        print("📊 Initializing report generator...")
        self.report_generator = ReportGenerator()

    async def run_comprehensive_evaluation(self, model_name: str = "grok-3"):
        """Run comprehensive multi-dimensional evaluation"""
        print(f"\n{'='*80}")
        print(f"🏆 STARTING COMPREHENSIVE EVALUATION: {model_name}")
        print(f"{'='*80}")

        try:
            # Run the comprehensive evaluation
            print("⏳ Running comprehensive evaluation suite...")
            print("   This includes:")
            print("   • 📚 Capability Assessment (Reasoning, Knowledge, Creativity)")
            print("   • 🛡️  Safety Evaluation (Toxicity, Bias, Hallucination)")
            print("   • 🧪 Robustness Testing (Adversarial, Edge cases, Injection)")
            print("   • 🔄 Consistency Analysis (Temporal, Cross-prompt, Factual)")
            print("   • 🎯 Behavioral Testing (Helpfulness, Harmlessness, Honesty)")
            print()

            result = await self.orchestrator.run_evaluation(self.model_provider, model_name)

            print(f"\n🎉 EVALUATION COMPLETED!")
            print(f"Overall Score: {result.overall_score:.1f}/100")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Risk Assessment: {result.risk_assessment}")
            print(f"Duration: {result.metadata['duration_seconds']:.1f} seconds")
            print(f"Total Tests: {result.metadata['total_tests']}")

            # Show module breakdown
            print(f"\n📈 Module Scores:")
            for module, score in result.evaluation_summary.items():
                status = "✅" if score >= 75 else "⚠️" if score >= 60 else "❌"
                print(f"   {status} {module.capitalize():15} {score:6.1f}/100")

            # Show recommendations
            if result.recommendations:
                print(f"\n💡 Top Recommendations:")
                for i, rec in enumerate(result.recommendations[:3], 1):
                    print(f"   {i}. {rec}")

            return result

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

    async def generate_reports(self, result, model_name: str):
        """Generate reports in all formats"""
        print(f"\n📄 Generating comprehensive reports...")

        try:
            # Generate all format reports
            report_files = self.report_generator.generate_all_formats(result, f"{model_name}_comprehensive")

            print("📋 Generated reports:")
            for format_type, filepath in report_files.items():
                print(f"   • {format_type.upper()}: {filepath}")

            return report_files

        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            raise

    async def run_module_tests(self):
        """Test individual modules"""
        print(f"\n{'='*80}")
        print("🔬 TESTING INDIVIDUAL MODULES")
        print(f"{'='*80}")

        try:
            # Test capability assessment
            print("\n📚 Testing Capability Assessment...")
            from src.evaluation.multidimensional import CapabilityAssessment
            capability_eval = CapabilityAssessment()
            capability_result = await capability_eval.run_capability_assessment(self.model_provider)
            print(f"   Capability Score: {capability_result.score:.1f}/100")

            # Test safety evaluation
            print("\n🛡️  Testing Safety Evaluation...")
            from src.evaluation.multidimensional import SafetyEvaluation
            safety_eval = SafetyEvaluation()
            safety_result = await safety_eval.run_safety_evaluation(self.model_provider)
            print(f"   Safety Score: {safety_result.safety_score:.1f}/100")

            # Test robustness testing
            print("\n🧪 Testing Robustness...")
            from src.evaluation.multidimensional import RobustnessTesting
            robustness_eval = RobustnessTesting()
            robustness_result = await robustness_eval.run_robustness_suite(self.model_provider)
            print(f"   Resilience Score: {robustness_result.resilience_score:.1f}/100")

            # Test consistency analysis
            print("\n🔄 Testing Consistency...")
            from src.evaluation.multidimensional import ConsistencyAnalysis
            consistency_eval = ConsistencyAnalysis()
            consistency_result = await consistency_eval.run_consistency_analysis(self.model_provider)
            print(f"   Consistency Score: {consistency_result.consistency_score:.1f}/100")

            # Test behavioral testing
            print("\n🎯 Testing Behavioral Assessment...")
            from src.evaluation.multidimensional import BehavioralTesting
            behavioral_eval = BehavioralTesting()
            behavioral_result = await behavioral_eval.run_behavioral_assessment(self.model_provider)
            print(f"   Behavioral Score: {behavioral_result.score:.1f}/100")

            print("\n✅ All individual module tests completed successfully!")

        except Exception as e:
            print(f"❌ Module testing failed: {e}")
            raise

    async def run_evaluation_suite_comparison(self):
        """Compare different evaluation suite configurations"""
        print(f"\n{'='*80}")
        print("⚖️  COMPARING EVALUATION SUITES")
        print(f"{'='*80}")

        suite_results = {}

        # Test minimal suite
        print("\n🔷 Testing MINIMAL suite (Capability + Safety)...")
        minimal_config = self.orchestrator.create_evaluation_config(
            suite_type=EvaluationSuite.MINIMAL,
            modules=["capability", "safety"]
        )
        minimal_orchestrator = EvaluationOrchestrator(minimal_config)
        minimal_result = await minimal_orchestrator.run_evaluation(self.model_provider, "grok-3-minimal")
        suite_results["minimal"] = minimal_result

        # Test standard suite
        print("\n🔶 Testing STANDARD suite (Most modules)...")
        standard_config = self.orchestrator.create_evaluation_config(
            suite_type=EvaluationSuite.STANDARD
        )
        standard_orchestrator = EvaluationOrchestrator(standard_config)
        standard_result = await standard_orchestrator.run_evaluation(self.model_provider, "grok-3-standard")
        suite_results["standard"] = standard_result

        # Show comparison
        print(f"\n📊 Suite Comparison Results:")
        print(f"{'Suite':<12} {'Score':<8} {'Duration':<10} {'Tests':<8} {'Risk':<10}")
        print("-" * 50)

        for suite_name, result in suite_results.items():
            print(f"{suite_name.upper():<12} "
                  f"{result.overall_score:<8.1f} "
                  f"{result.metadata['duration_seconds']:<10.1f} "
                  f"{result.metadata['total_tests']:<8} "
                  f"{result.risk_assessment:<10}")

        # Generate comparison report
        print(f"\n📈 Generating suite comparison report...")
        comparison_file = self.report_generator.generate_comparison_report(
            list(suite_results.values()),
            "suite_comparison_grok3"
        )
        print(f"   Comparison report: {comparison_file}")

        return suite_results


async def main():
    """Main execution function"""
    print("🌟 GDP-eval Multi-Dimensional Evaluation Framework")
    print("Enterprise-grade AI model evaluation matching OpenAI/Anthropic standards")
    print("Author: Nicolas Delrieu, AI Consultant (+855 92 332 554)")
    print()

    runner = MultidimensionalEvaluationRunner()

    try:
        # Initialize everything
        runner.initialize_providers()

        # Test connection
        print("🔗 Testing API connection...")
        test_response = await runner.model_provider.generate("Hello, this is a test.")
        print(f"✅ Connection successful! Response length: {len(test_response.text)} chars")

        # Option 1: Run comprehensive evaluation
        print("\n" + "="*80)
        print("Choose evaluation mode:")
        print("1. 🏆 Comprehensive Evaluation (All modules)")
        print("2. 🔬 Individual Module Testing")
        print("3. ⚖️  Suite Comparison")
        print("4. 🚀 All of the above")
        print("="*80)

        choice = input("Enter your choice (1-4) or press Enter for option 4: ").strip()
        if not choice:
            choice = "4"

        if choice in ["1", "4"]:
            # Run comprehensive evaluation
            result = await runner.run_comprehensive_evaluation("grok-3")
            await runner.generate_reports(result, "grok-3")

        if choice in ["2", "4"]:
            # Test individual modules
            await runner.run_module_tests()

        if choice in ["3", "4"]:
            # Compare evaluation suites
            await runner.run_evaluation_suite_comparison()

        print(f"\n🎉 All evaluations completed successfully!")
        print(f"📁 Check the 'results/reports/' directory for generated reports")

    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 GDP-eval evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())