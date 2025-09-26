#!/usr/bin/env python3
"""
Test Grok API with GDP Evaluation Framework
"""

import asyncio
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.providers.grok_provider import GrokProvider
from src.core.task_loader import TaskLoader, EvaluationTask
from src.core.evaluator import Evaluator
from src.core.grader import AutomatedGrader, BilingualGrader
from src.metrics.performance_metrics import PerformanceAnalyzer
from src.metrics.khmer_metrics import KhmerMetrics


async def test_grok_connection(api_key: str):
    """Test basic Grok API connection"""
    print("\n" + "="*60)
    print("Testing Grok API Connection")
    print("="*60)

    provider = GrokProvider(
        api_key=api_key,
        model="grok-3"
    )

    # Test connection
    print("\n1. Testing connection...")
    connected = await provider.test_connection()
    if connected:
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return False

    # Test simple generation
    print("\n2. Testing generation...")
    response = await provider.generate(
        prompt="Say 'Hello from Cambodia' in both English and Khmer",
        max_tokens=50,
        temperature=0
    )

    if response.success:
        print(f"✓ Generation successful")
        print(f"   Response: {response.response[:100]}...")
        print(f"   Latency: {response.latency_ms:.2f}ms")
        print(f"   Tokens: {response.tokens_used}")
    else:
        print(f"✗ Generation failed: {response.error}")
        return False

    return True


async def run_gdp_evaluation(api_key: str):
    """Run comprehensive GDP evaluation with Grok"""
    print("\n" + "="*60)
    print("Running GDP Evaluation with Grok")
    print("="*60)

    # Initialize provider
    provider = GrokProvider(
        api_key=api_key,
        model="grok-3"
    )

    # Create sample evaluation tasks
    tasks = [
        EvaluationTask(
            task_id="finance_001",
            category="finance",
            occupation="Loan Officer",
            prompt={
                "instruction": "Evaluate this loan application and provide a risk assessment",
                "context": "Applicant: Small business owner in Phnom Penh. Business: Restaurant for 3 years. Monthly revenue: $8,000. Requested loan: $20,000 for expansion. Collateral: Business equipment worth $15,000.",
                "requirements": ["Risk level (low/medium/high)", "Recommended interest rate", "Repayment period", "Key concerns"],
                "language": "bilingual"
            },
            grading_criteria=[
                {
                    "criterion_id": "accuracy",
                    "criterion_name": "Financial Analysis Accuracy",
                    "weight": 2.0,
                    "max_score": 10
                },
                {
                    "criterion_id": "completeness",
                    "criterion_name": "Complete Assessment",
                    "weight": 1.5,
                    "max_score": 10
                }
            ],
            metadata={
                "language": "bilingual",
                "complexity": 3,
                "time_limit_minutes": 10
            }
        ),
        EvaluationTask(
            task_id="agriculture_001",
            category="agriculture",
            occupation="Agricultural Advisor",
            prompt={
                "instruction": "ផ្តល់ដំបូន្មានអំពីការដាំស្រូវក្នុងរដូវវស្សា (Provide advice about rice planting in rainy season)",
                "context": "Location: Battambang province. Field size: 2 hectares. Soil: Clay loam. Previous crop: Cassava. Water source: Rainfall and small irrigation canal.",
                "requirements": ["Recommended rice variety", "Planting schedule", "Fertilizer plan", "Pest management", "Expected yield"],
                "language": "khmer_preferred"
            },
            grading_criteria=[
                {
                    "criterion_id": "technical_correctness",
                    "criterion_name": "Agricultural Knowledge",
                    "weight": 2.0,
                    "max_score": 10
                },
                {
                    "criterion_id": "language_quality",
                    "criterion_name": "Khmer Language Use",
                    "weight": 1.0,
                    "max_score": 10
                }
            ],
            metadata={
                "language": "khmer",
                "complexity": 3,
                "time_limit_minutes": 10
            }
        ),
        EvaluationTask(
            task_id="tourism_001",
            category="tourism",
            occupation="Tour Coordinator",
            prompt={
                "instruction": "Design a 3-day tour package for Siem Reap",
                "context": "Client: Family of 4 (2 adults, 2 children aged 10 and 14). Budget: $800 total. Interests: Temples, culture, local food. Hotel: Already booked near Pub Street.",
                "requirements": ["Day-by-day itinerary", "Entrance fees breakdown", "Transportation plan", "Meal recommendations", "Total cost estimate"],
                "language": "english"
            },
            grading_criteria=[
                {
                    "criterion_id": "completeness",
                    "criterion_name": "Itinerary Completeness",
                    "weight": 1.5,
                    "max_score": 10
                },
                {
                    "criterion_id": "accuracy",
                    "criterion_name": "Cost Accuracy",
                    "weight": 2.0,
                    "max_score": 10
                }
            ],
            metadata={
                "language": "english",
                "complexity": 2,
                "time_limit_minutes": 15
            }
        )
    ]

    # Initialize evaluator
    evaluator = Evaluator(
        provider=provider,
        output_dir="results/grok_test"
    )

    # Run evaluation
    print("\n3. Running evaluation tasks...")
    results = []
    for task in tasks:
        print(f"\n   Task: {task.task_id} ({task.occupation})")
        result = await evaluator.evaluate_single(task)
        results.append(result)

        if result.success:
            print(f"   ✓ Completed in {result.latency_ms:.2f}ms")
            print(f"   Response preview: {result.response[:100]}...")
        else:
            print(f"   ✗ Failed: {result.error}")

    # Grade results
    print("\n4. Grading responses...")
    grader = BilingualGrader()
    grading_results = []

    for task, result in zip(tasks, results):
        if result.success:
            grade = grader.grade_response(task, result)
            grading_results.append(grade)
            print(f"\n   {task.task_id}:")
            print(f"   Score: {grade.percentage:.1f}%")
            print(f"   Feedback: {list(grade.feedback.values())[0][:100]}...")

    # Analyze performance
    print("\n5. Performance Analysis...")
    analyzer = PerformanceAnalyzer()

    for result in results:
        analyzer.add_result(result)

    stats = analyzer.get_statistics()
    print(f"\n   Average latency: {stats['latency']['mean']:.2f}ms")
    print(f"   Total tokens: {stats['tokens']['total']}")
    print(f"   Success rate: {stats['success_rate']*100:.1f}%")

    # Khmer metrics analysis
    print("\n6. Khmer Language Analysis...")
    khmer_metrics = KhmerMetrics()

    for result in results:
        if result.success and result.metadata.get('language') in ['khmer', 'bilingual']:
            analysis = khmer_metrics.analyze_text(result.response)
            print(f"\n   {result.task_id}:")
            print(f"   Khmer ratio: {analysis['language_distribution'].get('khmer_ratio', 0)*100:.1f}%")
            print(f"   Code-switching: {'Yes' if analysis['code_switching']['has_code_switching'] else 'No'}")

    # Save results
    print("\n7. Saving results...")
    output_dir = "results/grok_test"
    os.makedirs(output_dir, exist_ok=True)

    # Save evaluation results
    with open(f"{output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [r.to_dict() for r in results],
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save grading results
    with open(f"{output_dir}/grading_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [g.to_dict() for g in grading_results],
            f,
            indent=2,
            ensure_ascii=False
        )

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "provider": "Grok (xAI)",
        "model": "grok-3",
        "tasks_evaluated": len(tasks),
        "success_rate": stats['success_rate'],
        "average_latency_ms": stats['latency']['mean'],
        "average_score": sum(g.percentage for g in grading_results) / len(grading_results) if grading_results else 0,
        "grading_summary": grader.get_statistics()
    }

    with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n   Results saved to {output_dir}/")

    return summary


async def main():
    """Main test function"""
    # API key from user
    API_KEY = "YOUR-API-KEY-HERE"

    print("\n" + "="*60)
    print("GDP-eval Framework - Grok API Test")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: Grok 3")

    # Test connection first
    connected = await test_grok_connection(API_KEY)

    if not connected:
        print("\n✗ Connection test failed. Please check your API key.")
        return

    # Run full evaluation
    try:
        summary = await run_gdp_evaluation(API_KEY)

        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        print(f"\n✓ Success rate: {summary['success_rate']*100:.1f}%")
        print(f"✓ Average score: {summary['average_score']:.1f}%")
        print(f"✓ Average latency: {summary['average_latency_ms']:.2f}ms")

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())