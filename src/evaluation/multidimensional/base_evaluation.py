"""
Base Evaluation Module - Common functionality for all evaluation modules
Handles dataset loading, test sampling, and common operations
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes"""
    QUICK = "quick"      # 10 tests per module for rapid assessment
    STANDARD = "standard"  # 30 tests per module for standard assessment
    FULL = "full"        # All available tests for comprehensive assessment


class BaseEvaluationModule:
    """
    Base class for all evaluation modules
    Provides common dataset loading and test sampling functionality
    """

    def __init__(self, dataset_name: str, test_mode: TestMode = TestMode.STANDARD):
        """
        Initialize base evaluation module

        Args:
            dataset_name: Name of the JSON dataset file (without path)
            test_mode: Test execution mode (quick/standard/full)
        """
        self.dataset_name = dataset_name
        self.test_mode = test_mode
        self.test_suite = self._load_dataset()
        self.sampled_tests = self._sample_tests()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load test dataset from JSON file"""
        try:
            # Try multiple paths to find the dataset
            base_paths = [
                Path(__file__).parent.parent.parent.parent,  # Project root
                Path.cwd()  # Current working directory
            ]

            for base_path in base_paths:
                test_file = base_path / "data" / "evaluation" / self.dataset_name

                if test_file.exists():
                    with open(test_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Handle different JSON structures
                        if isinstance(data, dict):
                            if 'tests' in data:
                                tests = data['tests']
                            elif 'test_cases' in data:
                                tests = data['test_cases']
                            elif 'items' in data:
                                tests = data['items']
                            else:
                                # Assume the dict values are the tests
                                tests = list(data.values()) if data else []
                        elif isinstance(data, list):
                            tests = data
                        else:
                            tests = []

                        logger.info(f"Loaded {len(tests)} tests from {self.dataset_name}")
                        return tests

            logger.warning(f"Dataset {self.dataset_name} not found in any expected location")

        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")

        # Return empty list if loading fails
        return []

    def _sample_tests(self) -> List[Dict[str, Any]]:
        """Sample tests based on test mode"""
        if not self.test_suite:
            return []

        # Determine sample size based on mode
        sample_sizes = {
            TestMode.QUICK: min(10, len(self.test_suite)),
            TestMode.STANDARD: min(30, len(self.test_suite)),
            TestMode.FULL: len(self.test_suite)
        }

        sample_size = sample_sizes.get(self.test_mode, 30)

        # Sample tests
        if sample_size >= len(self.test_suite):
            sampled = self.test_suite
        else:
            # Try to sample evenly across categories if they exist
            sampled = self._stratified_sample(sample_size)

        logger.info(f"Sampled {len(sampled)} tests in {self.test_mode.value} mode")
        return sampled

    def _stratified_sample(self, sample_size: int) -> List[Dict[str, Any]]:
        """
        Perform stratified sampling to get balanced test representation

        Args:
            sample_size: Number of tests to sample

        Returns:
            Sampled tests with balanced category representation
        """
        # Group tests by category if available
        categories = {}
        for test in self.test_suite:
            category = test.get('category', test.get('type', 'default'))
            if category not in categories:
                categories[category] = []
            categories[category].append(test)

        # If only one category, just random sample
        if len(categories) <= 1:
            return random.sample(self.test_suite, sample_size)

        # Calculate samples per category
        samples_per_category = max(1, sample_size // len(categories))
        remaining = sample_size - (samples_per_category * len(categories))

        sampled = []

        # Sample from each category
        for category, tests in categories.items():
            n_samples = min(samples_per_category, len(tests))
            sampled.extend(random.sample(tests, n_samples))

        # Add remaining samples randomly
        if remaining > 0:
            all_unsampled = [t for t in self.test_suite if t not in sampled]
            if all_unsampled:
                additional = random.sample(
                    all_unsampled,
                    min(remaining, len(all_unsampled))
                )
                sampled.extend(additional)

        return sampled

    def get_test_suite(self, use_sampling: bool = True) -> List[Dict[str, Any]]:
        """
        Get test suite for evaluation

        Args:
            use_sampling: Whether to use sampled tests or full suite

        Returns:
            List of tests to execute
        """
        if use_sampling and self.test_mode != TestMode.FULL:
            return self.sampled_tests
        return self.test_suite

    def get_test_count(self) -> Dict[str, int]:
        """Get test count statistics"""
        return {
            "total_available": len(self.test_suite),
            "sampled": len(self.sampled_tests),
            "mode": self.test_mode.value
        }

    @staticmethod
    def parse_test_mode(mode_str: Optional[str]) -> TestMode:
        """
        Parse test mode from string

        Args:
            mode_str: String representation of test mode

        Returns:
            TestMode enum value
        """
        if not mode_str:
            return TestMode.STANDARD

        mode_map = {
            "quick": TestMode.QUICK,
            "fast": TestMode.QUICK,
            "standard": TestMode.STANDARD,
            "normal": TestMode.STANDARD,
            "full": TestMode.FULL,
            "complete": TestMode.FULL,
            "comprehensive": TestMode.FULL
        }

        return mode_map.get(mode_str.lower(), TestMode.STANDARD)