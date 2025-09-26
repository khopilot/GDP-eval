"""
Performance Metrics Analyzer
Tracks and analyzes model performance metrics
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    task_id: str
    metric_type: str
    value: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceAnalyzer:
    """
    Analyzes performance metrics for model evaluations
    """

    def __init__(self):
        """Initialize performance analyzer"""
        self.metrics: List[PerformanceMetric] = []
        self.results = []
        self.start_time = time.time()

    def add_result(self, result: Any):
        """
        Add an evaluation result for analysis

        Args:
            result: Evaluation result object
        """
        self.results.append(result)

        # Extract metrics
        if hasattr(result, 'latency_ms'):
            self.add_metric(
                task_id=result.task_id,
                metric_type="latency",
                value=result.latency_ms
            )

        if hasattr(result, 'tokens_used'):
            self.add_metric(
                task_id=result.task_id,
                metric_type="tokens",
                value=result.tokens_used
            )

        if hasattr(result, 'api_cost'):
            self.add_metric(
                task_id=result.task_id,
                metric_type="cost",
                value=result.api_cost
            )

    def add_metric(
        self,
        task_id: str,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a performance metric

        Args:
            task_id: Task identifier
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            task_id=task_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.metrics.append(metric)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate performance statistics

        Returns:
            Dictionary with performance metrics
        """
        if not self.results:
            return {
                "total_tasks": 0,
                "success_rate": 0,
                "latency": {},
                "tokens": {},
                "throughput": {}
            }

        # Success rate
        successful = [r for r in self.results if getattr(r, 'success', False)]
        success_rate = len(successful) / len(self.results) if self.results else 0

        # Latency statistics
        latencies = [m.value for m in self.metrics if m.metric_type == "latency"]
        latency_stats = self._calculate_stats(latencies) if latencies else {}

        # Token statistics
        tokens = [m.value for m in self.metrics if m.metric_type == "tokens"]
        token_stats = self._calculate_stats(tokens) if tokens else {}

        # Cost statistics
        costs = [m.value for m in self.metrics if m.metric_type == "cost"]
        cost_stats = self._calculate_stats(costs) if costs else {}

        # Throughput calculation
        elapsed_time = time.time() - self.start_time
        throughput = {
            "tasks_per_second": len(self.results) / elapsed_time if elapsed_time > 0 else 0,
            "tokens_per_second": sum(tokens) / elapsed_time if elapsed_time > 0 and tokens else 0,
            "total_time_seconds": elapsed_time
        }

        return {
            "total_tasks": len(self.results),
            "successful_tasks": len(successful),
            "failed_tasks": len(self.results) - len(successful),
            "success_rate": success_rate,
            "latency": latency_stats,
            "tokens": token_stats,
            "cost": cost_stats,
            "throughput": throughput
        }

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate statistical metrics for a list of values

        Args:
            values: List of numerical values

        Returns:
            Statistics dictionary
        """
        if not values:
            return {}

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "total": float(np.sum(values))
        }

    def get_task_performance(self, task_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific task

        Args:
            task_id: Task identifier

        Returns:
            Task performance metrics
        """
        task_metrics = [m for m in self.metrics if m.task_id == task_id]

        if not task_metrics:
            return {}

        performance = {}
        for metric in task_metrics:
            performance[metric.metric_type] = metric.value

        # Get result details
        task_result = next((r for r in self.results if r.task_id == task_id), None)
        if task_result:
            performance["success"] = getattr(task_result, "success", False)
            performance["error"] = getattr(task_result, "error", None)

        return performance

    def get_category_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics by category

        Returns:
            Performance by category
        """
        category_metrics = {}

        for result in self.results:
            category = result.metadata.get("category", "unknown") if hasattr(result, "metadata") else "unknown"

            if category not in category_metrics:
                category_metrics[category] = {
                    "total": 0,
                    "successful": 0,
                    "latencies": [],
                    "tokens": [],
                    "costs": []
                }

            category_metrics[category]["total"] += 1

            if getattr(result, "success", False):
                category_metrics[category]["successful"] += 1

            if hasattr(result, "latency_ms"):
                category_metrics[category]["latencies"].append(result.latency_ms)

            if hasattr(result, "tokens_used"):
                category_metrics[category]["tokens"].append(result.tokens_used)

            if hasattr(result, "api_cost"):
                category_metrics[category]["costs"].append(result.api_cost)

        # Calculate statistics for each category
        for category, data in category_metrics.items():
            data["success_rate"] = data["successful"] / data["total"] if data["total"] > 0 else 0
            data["avg_latency"] = np.mean(data["latencies"]) if data["latencies"] else 0
            data["avg_tokens"] = np.mean(data["tokens"]) if data["tokens"] else 0
            data["total_cost"] = sum(data["costs"])

            # Remove raw lists
            del data["latencies"]
            del data["tokens"]
            del data["costs"]

        return category_metrics

    def get_bottlenecks(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks

        Args:
            threshold_ms: Latency threshold in milliseconds

        Returns:
            List of slow tasks
        """
        bottlenecks = []

        for result in self.results:
            if hasattr(result, "latency_ms") and result.latency_ms > threshold_ms:
                bottlenecks.append({
                    "task_id": result.task_id,
                    "latency_ms": result.latency_ms,
                    "category": result.metadata.get("category", "unknown") if hasattr(result, "metadata") else "unknown",
                    "tokens": getattr(result, "tokens_used", 0)
                })

        # Sort by latency
        bottlenecks.sort(key=lambda x: x["latency_ms"], reverse=True)

        return bottlenecks

    def get_efficiency_score(self) -> float:
        """
        Calculate enterprise-grade efficiency score with complexity awareness

        Returns:
            Efficiency score (0-100)
        """
        stats = self.get_statistics()

        if stats["total_tasks"] == 0:
            return 0

        # Dynamic weight calculation based on use case
        # For professional tasks, quality matters more than speed
        success_weight = 0.35  # Task completion
        quality_weight = 0.25  # Response quality
        latency_weight = 0.20  # Response time (adjusted for complexity)
        cost_weight = 0.10    # Cost efficiency
        consistency_weight = 0.10  # Consistency across tasks

        # Success score
        success_score = stats["success_rate"] * 100 * success_weight

        # Quality score (based on grading if available)
        quality_score = 0
        if hasattr(self, 'quality_scores') and self.quality_scores:
            avg_quality = sum(self.quality_scores) / len(self.quality_scores)
            quality_score = avg_quality * quality_weight
        else:
            # Use success rate as proxy for quality if no grading available
            quality_score = stats["success_rate"] * 100 * quality_weight

        # Latency score adjusted for task complexity
        if stats.get("latency") and stats["latency"].get("mean"):
            latency_mean = stats["latency"]["mean"]

            # Adjusted thresholds for AI tasks (more realistic)
            # Good: < 30s for complex tasks, < 10s for simple tasks
            # Bad: > 120s for complex, > 30s for simple
            complexity_factor = getattr(self, 'avg_complexity', 3) / 3  # Normalize complexity 1-5 to 0.33-1.67

            good_threshold = 10000 + (20000 * complexity_factor)  # 10-30s based on complexity
            bad_threshold = 30000 + (90000 * complexity_factor)   # 30-120s based on complexity

            if latency_mean < good_threshold:
                latency_score = 100 * latency_weight
            elif latency_mean > bad_threshold:
                latency_score = 0
            else:
                ratio = (latency_mean - good_threshold) / (bad_threshold - good_threshold)
                latency_score = (1 - ratio) * 100 * latency_weight
        else:
            latency_score = 50 * latency_weight  # Default middle score

        # Cost efficiency score
        cost_score = 0
        if stats.get("cost") and stats["cost"].get("mean"):
            # Assume $0.01 per task is good, $0.10 is bad
            cost_mean = stats["cost"]["mean"]
            if cost_mean < 0.01:
                cost_score = 100 * cost_weight
            elif cost_mean > 0.10:
                cost_score = 0
            else:
                cost_score = (1 - (cost_mean - 0.01) / 0.09) * 100 * cost_weight
        else:
            cost_score = 50 * cost_weight  # Default middle score

        # Consistency score (low variance is good)
        consistency_score = 0
        if stats.get("latency") and stats["latency"].get("std"):
            # Low standard deviation means consistent performance
            std_ratio = stats["latency"]["std"] / (stats["latency"]["mean"] + 1)
            if std_ratio < 0.2:  # Very consistent
                consistency_score = 100 * consistency_weight
            elif std_ratio < 0.5:  # Reasonably consistent
                consistency_score = 60 * consistency_weight
            else:  # Inconsistent
                consistency_score = 20 * consistency_weight
        else:
            consistency_score = 50 * consistency_weight

        total_score = success_score + quality_score + latency_score + cost_score + consistency_score
        return min(100, total_score)  # Cap at 100

    def export_metrics(self, filepath: str):
        """
        Export performance metrics to file

        Args:
            filepath: Path to save metrics
        """
        import json

        output = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "category_performance": self.get_category_performance(),
            "bottlenecks": self.get_bottlenecks(),
            "efficiency_score": self.get_efficiency_score(),
            "raw_metrics": [
                {
                    "task_id": m.task_id,
                    "type": m.metric_type,
                    "value": m.value,
                    "timestamp": m.timestamp
                }
                for m in self.metrics
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Performance metrics exported to {filepath}")