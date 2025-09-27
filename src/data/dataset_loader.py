"""
Dataset Loader - Enterprise Grade
Batch loading and caching for expanded GDP-eval datasets
Author: Nicolas Delrieu (+855 92 332 554)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Types of datasets available"""
    PROFESSIONAL_TASKS = "professional_tasks"
    CAPABILITY_TESTS = "capability_tests"
    SAFETY_TESTS = "safety_tests"
    ROBUSTNESS_TESTS = "robustness_tests"
    CONSISTENCY_TESTS = "consistency_tests"
    BEHAVIORAL_TESTS = "behavioral_tests"
    KHMER_TESTS = "khmer_tests"
    SECTOR_BENCHMARKS = "benchmarks"


@dataclass
class DatasetMetadata:
    """Metadata for loaded datasets"""
    name: str
    version: str
    test_count: int
    categories: Dict[str, int]
    language_distribution: Dict[str, float]
    complexity_distribution: Dict[int, int]
    file_path: str
    file_hash: str
    loaded_at: str


@dataclass
class DatasetFilter:
    """Filtering criteria for dataset loading"""
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    complexity_range: Optional[tuple] = None
    sectors: Optional[List[str]] = None
    max_items: Optional[int] = None
    random_sample: bool = False
    seed: Optional[int] = None


class DatasetLoader:
    """
    Enterprise-grade dataset loader with caching and filtering
    """

    def __init__(
        self,
        base_path: str = "data",
        cache_enabled: bool = True,
        lazy_load: bool = True
    ):
        """
        Initialize dataset loader

        Args:
            base_path: Base directory for datasets
            cache_enabled: Enable in-memory caching
            lazy_load: Load datasets only when accessed
        """
        self.base_path = Path(base_path)
        self.cache_enabled = cache_enabled
        self.lazy_load = lazy_load
        self._cache: Dict[str, Any] = {}
        self._metadata: Dict[str, DatasetMetadata] = {}

        # Dataset paths configuration
        self.dataset_paths = {
            DatasetType.PROFESSIONAL_TASKS: "tasks/professional_tasks_expanded.json",
            DatasetType.CAPABILITY_TESTS: "evaluation/capability_tests.json",
            DatasetType.SAFETY_TESTS: "evaluation/safety_tests.json",
            DatasetType.ROBUSTNESS_TESTS: "evaluation/robustness_tests.json",
            DatasetType.CONSISTENCY_TESTS: "evaluation/consistency_tests.json",
            DatasetType.BEHAVIORAL_TESTS: "evaluation/behavioral_tests.json",
            DatasetType.KHMER_TESTS: "khmer/evaluation_suite.json",
            DatasetType.SECTOR_BENCHMARKS: "benchmarks/finance_benchmark.json"
        }

        if not lazy_load:
            self._preload_all()

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for cache invalidation"""
        if not file_path.exists():
            return ""

        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _preload_all(self):
        """Preload all available datasets"""
        for dataset_type in DatasetType:
            try:
                self.load_dataset(dataset_type)
            except FileNotFoundError:
                logger.warning(f"Dataset {dataset_type.value} not found")

    @lru_cache(maxsize=32)
    def load_dataset(
        self,
        dataset_type: DatasetType,
        filter_criteria: Optional[DatasetFilter] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Load a dataset with optional filtering

        Args:
            dataset_type: Type of dataset to load
            filter_criteria: Optional filtering criteria
            force_reload: Force reload from disk

        Returns:
            Loaded and filtered dataset
        """
        cache_key = f"{dataset_type.value}_{hash(str(filter_criteria))}"

        # Check cache
        if self.cache_enabled and not force_reload:
            if cache_key in self._cache:
                logger.debug(f"Loading {dataset_type.value} from cache")
                return self._cache[cache_key]

        # Load from disk
        file_path = self.base_path / self.dataset_paths[dataset_type]

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading dataset from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract metadata
        self._extract_metadata(dataset_type, data, file_path)

        # Apply filters
        if filter_criteria:
            data = self._apply_filters(data, filter_criteria)

        # Cache if enabled
        if self.cache_enabled:
            self._cache[cache_key] = data

        return data

    def _extract_metadata(
        self,
        dataset_type: DatasetType,
        data: Dict[str, Any],
        file_path: Path
    ):
        """Extract and store dataset metadata"""
        from datetime import datetime

        metadata = DatasetMetadata(
            name=dataset_type.value,
            version=data.get("metadata", {}).get("version", "1.0.0"),
            test_count=data.get("metadata", {}).get("test_count", 0),
            categories=self._count_categories(data),
            language_distribution=self._analyze_languages(data),
            complexity_distribution=self._analyze_complexity(data),
            file_path=str(file_path),
            file_hash=self._get_file_hash(file_path),
            loaded_at=datetime.now().isoformat()
        )

        self._metadata[dataset_type.value] = metadata

    def _count_categories(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Count items per category"""
        categories = {}

        # Handle different data structures
        if "test_categories" in data:
            for category, subcats in data["test_categories"].items():
                if isinstance(subcats, dict):
                    categories[category] = sum(subcats.values())
                else:
                    categories[category] = subcats

        elif "tasks" in data:
            for task in data["tasks"]:
                sector = task.get("sector", "unknown")
                categories[sector] = categories.get(sector, 0) + 1

        elif "tests" in data:
            for test in data["tests"]:
                category = test.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1

        return categories

    def _analyze_languages(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze language distribution"""
        languages = {}
        total = 0

        items = data.get("tasks") or data.get("tests") or []

        for item in items:
            lang = item.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            total += 1

        # Convert to percentages
        if total > 0:
            languages = {k: (v/total)*100 for k, v in languages.items()}

        return languages

    def _analyze_complexity(self, data: Dict[str, Any]) -> Dict[int, int]:
        """Analyze complexity distribution"""
        complexity = {}

        items = data.get("tasks") or data.get("tests") or []

        for item in items:
            level = item.get("complexity", 0)
            complexity[level] = complexity.get(level, 0) + 1

        return complexity

    def _apply_filters(
        self,
        data: Dict[str, Any],
        filter_criteria: DatasetFilter
    ) -> Dict[str, Any]:
        """Apply filtering criteria to dataset"""
        import random

        filtered_data = data.copy()
        items_key = "tasks" if "tasks" in data else "tests"
        items = data.get(items_key, [])

        # Apply category filter
        if filter_criteria.categories:
            items = [
                item for item in items
                if item.get("category") in filter_criteria.categories
                or item.get("sector") in filter_criteria.categories
            ]

        # Apply subcategory filter
        if filter_criteria.subcategories:
            items = [
                item for item in items
                if item.get("subcategory") in filter_criteria.subcategories
            ]

        # Apply language filter
        if filter_criteria.languages:
            items = [
                item for item in items
                if item.get("language") in filter_criteria.languages
            ]

        # Apply complexity filter
        if filter_criteria.complexity_range:
            min_c, max_c = filter_criteria.complexity_range
            items = [
                item for item in items
                if min_c <= item.get("complexity", 0) <= max_c
            ]

        # Apply sector filter
        if filter_criteria.sectors:
            items = [
                item for item in items
                if item.get("sector") in filter_criteria.sectors
            ]

        # Apply random sampling
        if filter_criteria.random_sample and filter_criteria.max_items:
            if filter_criteria.seed:
                random.seed(filter_criteria.seed)

            if len(items) > filter_criteria.max_items:
                items = random.sample(items, filter_criteria.max_items)

        # Apply max items limit
        elif filter_criteria.max_items:
            items = items[:filter_criteria.max_items]

        filtered_data[items_key] = items

        # Update metadata
        if "metadata" in filtered_data:
            filtered_data["metadata"]["filtered"] = True
            filtered_data["metadata"]["original_count"] = len(data.get(items_key, []))
            filtered_data["metadata"]["filtered_count"] = len(items)

        return filtered_data

    def load_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Load all available datasets"""
        all_data = {}

        for dataset_type in DatasetType:
            try:
                all_data[dataset_type.value] = self.load_dataset(dataset_type)
                logger.info(f"Loaded {dataset_type.value}")
            except FileNotFoundError:
                logger.warning(f"Dataset {dataset_type.value} not found")

        return all_data

    def load_professional_tasks(
        self,
        sectors: Optional[List[str]] = None,
        complexity: Optional[tuple] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Convenience method to load professional tasks"""
        filter_criteria = DatasetFilter(
            sectors=sectors,
            complexity_range=complexity,
            max_items=limit
        )

        data = self.load_dataset(DatasetType.PROFESSIONAL_TASKS, filter_criteria)
        return data.get("tasks", [])

    def load_evaluation_tests(
        self,
        test_types: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load evaluation test suites"""
        evaluation_tests = {}

        test_datasets = [
            DatasetType.CAPABILITY_TESTS,
            DatasetType.SAFETY_TESTS,
            DatasetType.ROBUSTNESS_TESTS
        ]

        if test_types:
            test_datasets = [
                dt for dt in test_datasets
                if any(tt in dt.value for tt in test_types)
            ]

        for dataset_type in test_datasets:
            try:
                filter_criteria = DatasetFilter(
                    languages=languages,
                    max_items=limit
                )
                data = self.load_dataset(dataset_type, filter_criteria)
                evaluation_tests[dataset_type.value] = data.get("tests", [])
            except FileNotFoundError:
                logger.warning(f"Test suite {dataset_type.value} not found")

        return evaluation_tests

    def get_metadata(self, dataset_type: Optional[DatasetType] = None) -> Union[DatasetMetadata, Dict[str, DatasetMetadata]]:
        """Get metadata for datasets"""
        if dataset_type:
            return self._metadata.get(dataset_type.value)
        return self._metadata

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all datasets"""
        stats = {
            "total_items": 0,
            "datasets_loaded": len(self._metadata),
            "language_distribution": {},
            "complexity_distribution": {},
            "category_distribution": {},
            "cache_size": len(self._cache)
        }

        for metadata in self._metadata.values():
            stats["total_items"] += metadata.test_count

            # Aggregate language distribution
            for lang, pct in metadata.language_distribution.items():
                if lang not in stats["language_distribution"]:
                    stats["language_distribution"][lang] = 0
                stats["language_distribution"][lang] += pct * metadata.test_count / 100

            # Aggregate complexity distribution
            for level, count in metadata.complexity_distribution.items():
                if level not in stats["complexity_distribution"]:
                    stats["complexity_distribution"][level] = 0
                stats["complexity_distribution"][level] += count

            # Aggregate category distribution
            for category, count in metadata.categories.items():
                if category not in stats["category_distribution"]:
                    stats["category_distribution"][category] = 0
                stats["category_distribution"][category] += count

        return stats

    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()
        logger.info("Cache cleared")

    def validate_datasets(self) -> Dict[str, bool]:
        """Validate all dataset files"""
        validation_results = {}

        for dataset_type in DatasetType:
            file_path = self.base_path / self.dataset_paths[dataset_type]

            try:
                # Check file exists
                if not file_path.exists():
                    validation_results[dataset_type.value] = False
                    continue

                # Try to load and parse
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check basic structure
                has_metadata = "metadata" in data
                has_content = "tasks" in data or "tests" in data

                validation_results[dataset_type.value] = has_metadata and has_content

            except Exception as e:
                logger.error(f"Validation failed for {dataset_type.value}: {e}")
                validation_results[dataset_type.value] = False

        return validation_results


# Convenience functions
def load_all_datasets(base_path: str = "data") -> Dict[str, Dict[str, Any]]:
    """Quick function to load all datasets"""
    loader = DatasetLoader(base_path)
    return loader.load_all_datasets()


def load_filtered_tests(
    test_type: str,
    language: str = None,
    complexity: tuple = None,
    limit: int = None,
    base_path: str = "data"
) -> List[Dict[str, Any]]:
    """Load filtered tests by type"""
    loader = DatasetLoader(base_path)

    dataset_map = {
        "capability": DatasetType.CAPABILITY_TESTS,
        "safety": DatasetType.SAFETY_TESTS,
        "robustness": DatasetType.ROBUSTNESS_TESTS
    }

    dataset_type = dataset_map.get(test_type.lower())
    if not dataset_type:
        raise ValueError(f"Unknown test type: {test_type}")

    filter_criteria = DatasetFilter(
        languages=[language] if language else None,
        complexity_range=complexity,
        max_items=limit
    )

    data = loader.load_dataset(dataset_type, filter_criteria)
    return data.get("tests", [])


def get_dataset_statistics(base_path: str = "data") -> Dict[str, Any]:
    """Get statistics for all datasets"""
    loader = DatasetLoader(base_path)
    loader.load_all_datasets()  # Load to populate metadata
    return loader.get_statistics()


if __name__ == "__main__":
    # Example usage
    loader = DatasetLoader()

    # Load all datasets
    print("Loading all datasets...")
    all_data = loader.load_all_datasets()

    # Get statistics
    stats = loader.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"Total items: {stats['total_items']}")
    print(f"Datasets loaded: {stats['datasets_loaded']}")

    # Load filtered professional tasks
    print("\nLoading finance tasks...")
    finance_tasks = loader.load_professional_tasks(
        sectors=["finance"],
        complexity=(3, 5),
        limit=10
    )
    print(f"Loaded {len(finance_tasks)} finance tasks")

    # Validate all datasets
    print("\nValidating datasets...")
    validation = loader.validate_datasets()
    for dataset, valid in validation.items():
        status = "✓" if valid else "✗"
        print(f"{status} {dataset}")