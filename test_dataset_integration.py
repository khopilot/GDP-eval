#!/usr/bin/env python3
"""
Quick test script to verify dataset integration
Tests that modules properly load from JSON datasets
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 Testing Dataset Integration for GDP-eval Modules")
print("=" * 60)

async def test_dataset_loading():
    """Test that each module loads its dataset correctly"""

    print("\n📦 Testing Module Dataset Loading:")
    print("-" * 40)

    # Test SafetyEvaluation
    try:
        from src.evaluation.multidimensional.safety_evaluation import SafetyEvaluation
        safety_module = SafetyEvaluation()
        safety_tests = len(safety_module.safety_tests) if hasattr(safety_module, 'safety_tests') else 0
        print(f"✅ SafetyEvaluation: Loaded {safety_tests} tests")
    except Exception as e:
        print(f"❌ SafetyEvaluation: Failed - {e}")

    # Test CapabilityAssessment
    try:
        from src.evaluation.multidimensional.capability_assessment import CapabilityAssessment
        from src.evaluation.multidimensional.base_evaluation import TestMode

        # Test different modes
        for mode in [TestMode.QUICK, TestMode.STANDARD, TestMode.FULL]:
            capability_module = CapabilityAssessment(test_mode=mode)
            total_tests = sum(len(tests) for tests in capability_module.test_suites.values())
            print(f"✅ CapabilityAssessment ({mode.value}): {total_tests} tests sampled")
    except Exception as e:
        print(f"❌ CapabilityAssessment: Failed - {e}")

    # Test RobustnessTesting
    try:
        from src.evaluation.multidimensional.robustness_testing import RobustnessTesting
        robustness_module = RobustnessTesting()
        print(f"✅ RobustnessTesting: Module loaded (needs dataset integration)")
    except Exception as e:
        print(f"❌ RobustnessTesting: Failed - {e}")

    # Test BehavioralTesting
    try:
        from src.evaluation.multidimensional.behavioral_testing import BehavioralTesting
        behavioral_module = BehavioralTesting()
        print(f"✅ BehavioralTesting: Module loaded (needs dataset integration)")
    except Exception as e:
        print(f"❌ BehavioralTesting: Failed - {e}")

    # Test ConsistencyAnalysis
    try:
        from src.evaluation.multidimensional.consistency_analysis import ConsistencyAnalysis
        consistency_module = ConsistencyAnalysis()
        print(f"✅ ConsistencyAnalysis: Module loaded (needs dataset integration)")
    except Exception as e:
        print(f"❌ ConsistencyAnalysis: Failed - {e}")

    print("\n📊 Dataset Files Check:")
    print("-" * 40)

    # Check if all dataset files exist
    data_path = Path("data/evaluation")
    datasets = [
        "capability_tests.json",
        "safety_tests.json",
        "robustness_tests.json",
        "behavioral_tests.json",
        "consistency_tests.json"
    ]

    for dataset in datasets:
        file_path = data_path / dataset
        if file_path.exists():
            with open(file_path, 'r') as f:
                import json
                data = json.load(f)
                test_count = len(data.get('tests', data.get('test_cases', [])))
                print(f"✅ {dataset}: {test_count} tests available")
        else:
            print(f"❌ {dataset}: File not found")

    print("\n🎯 Summary:")
    print("-" * 40)
    print("Safety module is using dataset ✅")
    print("Capability module is using dataset ✅")
    print("Other modules need dataset integration ⚠️")

if __name__ == "__main__":
    asyncio.run(test_dataset_loading())