#!/usr/bin/env python3
"""
Quick test with small subset to verify fixes before full workflow
Tests 2 tests per module = 10 total tests
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_small_subset():
    """Test with 2 tests per module to verify fixes"""

    print("🔧 TESTING FIXES WITH SMALL SUBSET")
    print("=" * 50)

    # Check API key
    api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ No API key found!")
        return

    # Initialize provider with validation
    from src.providers.grok_provider import GrokProvider
    provider = GrokProvider(api_key=api_key, model="grok-3", timeout=120)

    print("✅ Testing prompt validation...")
    test_prompt = "This is a test\x00\x1f with control chars"
    clean_prompt = provider._validate_and_sanitize_prompt(test_prompt)
    print(f"   Original: {len(test_prompt)} chars, Cleaned: {len(clean_prompt)} chars")

    # Test each module with 2 tests each
    results = {}

    print("\n🧪 TESTING INDIVIDUAL MODULES:")
    print("-" * 40)

    # Test SafetyEvaluation (2 tests)
    try:
        print("1. Testing SafetyEvaluation...")
        from src.evaluation.multidimensional.safety_evaluation import SafetyEvaluation
        safety = SafetyEvaluation()
        print(f"   ✅ Loaded {len(safety.safety_tests)} tests from dataset")
        print(f"   ✅ Has run_safety_evaluation method: {hasattr(safety, 'run_safety_evaluation')}")

        # Run 2 safety tests
        test_results = []
        for i, test in enumerate(safety.safety_tests[:2]):
            try:
                prompt = test.get('prompt', 'Test safety prompt')
                response = await provider.generate(prompt, max_tokens=50)
                test_results.append({"success": True, "text_length": len(response.text)})
                print(f"   ✅ Safety test {i+1}: {len(response.text)} chars")
            except Exception as e:
                test_results.append({"success": False, "error": str(e)})
                print(f"   ❌ Safety test {i+1}: {str(e)[:50]}...")

        results["safety"] = {
            "tests_run": 2,
            "success_count": sum(1 for r in test_results if r["success"]),
            "details": test_results
        }
    except Exception as e:
        print(f"   ❌ SafetyEvaluation failed: {e}")
        results["safety"] = {"error": str(e)}

    # Test RobustnessTesting (2 tests)
    try:
        print("\n2. Testing RobustnessTesting...")
        from src.evaluation.multidimensional.robustness_testing import RobustnessTesting
        robustness = RobustnessTesting()
        print(f"   ✅ Loaded {len(robustness.robustness_tests)} tests from dataset")
        print(f"   ✅ Has run_robustness_suite method: {hasattr(robustness, 'run_robustness_suite')}")

        # Run 2 robustness tests
        test_results = []
        for i, test in enumerate(robustness.robustness_tests[:2]):
            try:
                prompt = test.get('prompt', test.get('input', 'Test robustness prompt'))
                response = await provider.generate(prompt, max_tokens=50)
                test_results.append({"success": True, "text_length": len(response.text)})
                print(f"   ✅ Robustness test {i+1}: {len(response.text)} chars")
            except Exception as e:
                test_results.append({"success": False, "error": str(e)})
                print(f"   ❌ Robustness test {i+1}: {str(e)[:50]}...")

        results["robustness"] = {
            "tests_run": 2,
            "success_count": sum(1 for r in test_results if r["success"]),
            "details": test_results
        }
    except Exception as e:
        print(f"   ❌ RobustnessTesting failed: {e}")
        results["robustness"] = {"error": str(e)}

    # Test other modules (1 test each)
    module_tests = [
        ("capability", "CapabilityAssessment", "capability_tests"),
        ("behavioral", "BehavioralTesting", "behavioral_tests"),
        ("consistency", "ConsistencyAnalysis", "consistency_tests")
    ]

    for module_name, class_name, test_attr in module_tests:
        try:
            print(f"\n3. Testing {class_name}...")
            if module_name == "capability":
                module_path = f"src.evaluation.multidimensional.capability_assessment"
            elif module_name in ['behavioral', 'robustness']:
                module_path = f"src.evaluation.multidimensional.{module_name}_testing"
            else:
                module_path = f"src.evaluation.multidimensional.{module_name}_analysis"
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            tests = getattr(instance, test_attr, [])
            print(f"   ✅ Loaded {len(tests)} tests from dataset")

            # Run 1 test
            if tests:
                test = tests[0]
                prompt = test.get('prompt', test.get('input', f'Test {module_name} prompt'))
                try:
                    response = await provider.generate(prompt, max_tokens=50)
                    print(f"   ✅ {class_name} test: {len(response.text)} chars")
                    results[module_name] = {"tests_run": 1, "success_count": 1}
                except Exception as e:
                    print(f"   ❌ {class_name} test: {str(e)[:50]}...")
                    results[module_name] = {"tests_run": 1, "success_count": 0, "error": str(e)}
            else:
                print(f"   ⚠️ No tests found for {class_name}")
                results[module_name] = {"tests_run": 0, "success_count": 0, "error": "No tests"}
        except Exception as e:
            print(f"   ❌ {class_name} failed: {e}")
            results[module_name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("📊 SMALL SUBSET TEST RESULTS:")
    print("=" * 50)

    total_tests = 0
    total_success = 0

    for module, result in results.items():
        if "error" in result:
            print(f"❌ {module}: ERROR - {result['error'][:60]}...")
        else:
            tests_run = result.get("tests_run", 0)
            success_count = result.get("success_count", 0)
            total_tests += tests_run
            total_success += success_count
            status = "✅" if success_count == tests_run else "⚠️"
            print(f"{status} {module}: {success_count}/{tests_run} tests passed")

    print(f"\n🎯 OVERALL: {total_success}/{total_tests} tests passed")

    if total_success >= 8:  # At least 80% success
        print("✅ FIXES WORKING! Ready for full workflow")
        return True
    else:
        print("❌ ISSUES REMAIN! Check errors above")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_small_subset())
    sys.exit(0 if success else 1)