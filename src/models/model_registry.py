"""
Model Registry System
Central registry for managing fine-tuned models
"""

import os
import json
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import hashlib
import logging
from contextlib import contextmanager
import yaml

from src.models.model_metadata import (
    ModelMetadata, ModelTask, ModelFramework, QuantizationType,
    DeploymentStatus, TrainingConfig, DatasetInfo, PerformanceMetrics,
    HardwareRequirements, KhmerCapabilities
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for model management"""

    def __init__(self, registry_path: str = "models/registry", db_path: Optional[str] = None):
        """
        Initialize model registry

        Args:
            registry_path: Base path for model storage
            db_path: Path to SQLite database (default: registry_path/registry.db)
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Database path
        self.db_path = db_path or str(self.registry_path / "registry.db")

        # Initialize database
        self._init_database()

        # Model cache
        self._cache: Dict[str, ModelMetadata] = {}

    def _init_database(self):
        """Initialize SQLite database"""
        with self._get_db() as conn:
            cursor = conn.cursor()

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    task TEXT,
                    framework TEXT,
                    deployment_status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata_json TEXT,
                    UNIQUE(name, version)
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)

            # Deployments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    environment TEXT,
                    endpoint TEXT,
                    deployed_at TEXT,
                    status TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)

            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    dataset TEXT,
                    config_json TEXT,
                    metrics_json TEXT,
                    created_at TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models(deployment_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deployments_model ON deployments(model_id)")

            conn.commit()

    @contextmanager
    def _get_db(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def register(
        self,
        model_path: str,
        name: str,
        version: str,
        task: Union[str, ModelTask],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Register a new model

        Args:
            model_path: Path to model files
            name: Model name
            version: Model version
            task: Model task type
            metadata: Additional metadata
            **kwargs: Additional ModelMetadata fields

        Returns:
            Model ID
        """
        # Generate model ID
        model_id = self._generate_model_id(name, version)

        # Parse task
        if isinstance(task, str):
            task = ModelTask(task)

        # Create metadata object
        model_metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            task=task,
            **kwargs
        )

        # Add additional metadata
        if metadata:
            model_metadata.metadata.update(metadata)

        # Copy model files to registry
        model_registry_path = self.registry_path / model_id
        if Path(model_path).exists():
            logger.info(f"Copying model files from {model_path} to {model_registry_path}")
            if model_registry_path.exists():
                shutil.rmtree(model_registry_path)
            shutil.copytree(model_path, model_registry_path)

            # Calculate model size
            model_size_bytes = sum(
                f.stat().st_size for f in model_registry_path.rglob("*") if f.is_file()
            )
            model_metadata.model_size_gb = model_size_bytes / (1024**3)

        # Save metadata to file
        metadata_path = model_registry_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            f.write(model_metadata.to_json())

        # Save to database
        self._save_to_database(model_metadata)

        # Update cache
        self._cache[model_id] = model_metadata

        logger.info(f"Registered model: {model_id}")
        return model_id

    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        # Clean name and version
        clean_name = name.lower().replace(" ", "-").replace("_", "-")
        clean_version = version.lower().replace(" ", "-")
        return f"{clean_name}-{clean_version}"

    def _save_to_database(self, metadata: ModelMetadata) -> None:
        """Save model metadata to database"""
        with self._get_db() as conn:
            cursor = conn.cursor()

            # Check if model exists
            cursor.execute(
                "SELECT model_id FROM models WHERE model_id = ?",
                (metadata.model_id,)
            )
            exists = cursor.fetchone() is not None

            metadata_json = metadata.to_json()

            if exists:
                # Update existing
                cursor.execute("""
                    UPDATE models
                    SET name = ?, version = ?, task = ?, framework = ?,
                        deployment_status = ?, updated_at = ?, metadata_json = ?
                    WHERE model_id = ?
                """, (
                    metadata.name,
                    metadata.version,
                    metadata.task.value if metadata.task else None,
                    metadata.framework.value if metadata.framework else None,
                    metadata.deployment_status.value if metadata.deployment_status else None,
                    metadata.updated_at.isoformat(),
                    metadata_json,
                    metadata.model_id
                ))
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO models
                    (model_id, name, version, task, framework, deployment_status,
                     created_at, updated_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.name,
                    metadata.version,
                    metadata.task.value if metadata.task else None,
                    metadata.framework.value if metadata.framework else None,
                    metadata.deployment_status.value if metadata.deployment_status else None,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                    metadata_json
                ))

            # Save metrics if present
            if metadata.performance_metrics:
                self._save_metrics(cursor, metadata.model_id, metadata.performance_metrics)

            conn.commit()

    def _save_metrics(self, cursor, model_id: str, metrics: PerformanceMetrics) -> None:
        """Save performance metrics to database"""
        timestamp = datetime.now().isoformat()

        # Save each metric
        metric_values = {
            "perplexity": metrics.perplexity,
            "accuracy": metrics.accuracy,
            "khmer_bleu": metrics.khmer_bleu,
            "latency_ms_p50": metrics.latency_ms_p50,
            "memory_usage_gb": metrics.memory_usage_gb,
        }

        for metric_name, metric_value in metric_values.items():
            if metric_value is not None:
                cursor.execute("""
                    INSERT INTO metrics (model_id, metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (model_id, metric_name, metric_value, timestamp))

        # Save task-specific metrics
        for metric_name, metric_value in metrics.task_metrics.items():
            cursor.execute("""
                INSERT INTO metrics (model_id, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            """, (model_id, metric_name, metric_value, timestamp))

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model by ID

        Args:
            model_id: Model identifier

        Returns:
            Model metadata or None if not found
        """
        # Check cache
        if model_id in self._cache:
            return self._cache[model_id]

        # Load from database
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metadata_json FROM models WHERE model_id = ?",
                (model_id,)
            )
            result = cursor.fetchone()

            if result:
                metadata = ModelMetadata.from_json(result[0])
                self._cache[model_id] = metadata
                return metadata

        return None

    def list_models(
        self,
        task: Optional[ModelTask] = None,
        status: Optional[DeploymentStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filters

        Args:
            task: Filter by task type
            status: Filter by deployment status
            tags: Filter by tags

        Returns:
            List of model metadata
        """
        query = "SELECT metadata_json FROM models WHERE 1=1"
        params = []

        if task:
            query += " AND task = ?"
            params.append(task.value if isinstance(task, ModelTask) else task)

        if status:
            query += " AND deployment_status = ?"
            params.append(status.value if isinstance(status, DeploymentStatus) else status)

        query += " ORDER BY updated_at DESC"

        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()

        models = []
        for result in results:
            metadata = ModelMetadata.from_json(result[0])

            # Filter by tags if specified
            if tags:
                if not any(tag in metadata.tags for tag in tags):
                    continue

            models.append(metadata)

        return models

    def get_best_model(
        self,
        task: Union[str, ModelTask],
        metric: str = "accuracy",
        min_status: DeploymentStatus = DeploymentStatus.TESTING
    ) -> Optional[ModelMetadata]:
        """
        Get best model for a task based on metric

        Args:
            task: Task type
            metric: Metric to optimize
            min_status: Minimum deployment status

        Returns:
            Best model metadata or None
        """
        if isinstance(task, str):
            task = ModelTask(task)

        # Query for models with the task
        query = """
            SELECT m.metadata_json, MAX(met.metric_value) as best_metric
            FROM models m
            LEFT JOIN metrics met ON m.model_id = met.model_id
            WHERE m.task = ? AND met.metric_name = ?
            GROUP BY m.model_id
            ORDER BY best_metric DESC
            LIMIT 1
        """

        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (task.value, metric))
            result = cursor.fetchone()

            if result:
                metadata = ModelMetadata.from_json(result[0])

                # Check deployment status
                if metadata.deployment_status.value >= min_status.value:
                    return metadata

        return None

    def compare_versions(
        self,
        model_id1: str,
        model_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions

        Args:
            model_id1: First model ID
            model_id2: Second model ID

        Returns:
            Comparison results
        """
        model1 = self.get_model(model_id1)
        model2 = self.get_model(model_id2)

        if not model1 or not model2:
            raise ValueError("One or both models not found")

        comparison = {
            "model1": {
                "id": model1.model_id,
                "name": model1.name,
                "version": model1.version,
            },
            "model2": {
                "id": model2.model_id,
                "name": model2.name,
                "version": model2.version,
            },
            "metrics_comparison": {},
            "improvements": {},
            "regressions": {},
        }

        # Compare metrics
        if model1.performance_metrics and model2.performance_metrics:
            metrics1 = model1.performance_metrics
            metrics2 = model2.performance_metrics

            # Compare each metric
            for attr in ["perplexity", "accuracy", "khmer_bleu", "latency_ms_p50"]:
                val1 = getattr(metrics1, attr, None)
                val2 = getattr(metrics2, attr, None)

                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else 0

                    comparison["metrics_comparison"][attr] = {
                        "model1": val1,
                        "model2": val2,
                        "difference": diff,
                        "pct_change": pct_change
                    }

                    # Categorize changes
                    if attr in ["accuracy", "khmer_bleu"]:  # Higher is better
                        if diff > 0:
                            comparison["improvements"][attr] = pct_change
                        elif diff < 0:
                            comparison["regressions"][attr] = pct_change
                    elif attr in ["perplexity", "latency_ms_p50"]:  # Lower is better
                        if diff < 0:
                            comparison["improvements"][attr] = -pct_change
                        elif diff > 0:
                            comparison["regressions"][attr] = pct_change

        # Compare hardware requirements
        if model1.hardware_requirements and model2.hardware_requirements:
            comparison["hardware_comparison"] = {
                "gpu_memory": {
                    "model1": model1.hardware_requirements.min_gpu_memory_gb,
                    "model2": model2.hardware_requirements.min_gpu_memory_gb,
                },
                "ram": {
                    "model1": model1.hardware_requirements.min_ram_gb,
                    "model2": model2.hardware_requirements.min_ram_gb,
                }
            }

        return comparison

    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Update model metrics

        Args:
            model_id: Model identifier
            metrics: Dictionary of metric values
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        model.update_metrics(metrics)
        self._save_to_database(model)

        logger.info(f"Updated metrics for model {model_id}")

    def deploy_model(
        self,
        model_id: str,
        environment: str,
        endpoint: str,
        status: DeploymentStatus = DeploymentStatus.PRODUCTION
    ) -> None:
        """
        Deploy model to environment

        Args:
            model_id: Model identifier
            environment: Deployment environment
            endpoint: Serving endpoint
            status: Deployment status
        """
        model = self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Update model status
        model.set_deployment_status(status)
        model.add_endpoint(endpoint)

        # Save to database
        self._save_to_database(model)

        # Record deployment
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO deployments (model_id, environment, endpoint, deployed_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_id,
                environment,
                endpoint,
                datetime.now().isoformat(),
                status.value
            ))
            conn.commit()

        logger.info(f"Deployed model {model_id} to {environment} at {endpoint}")

    def get_deployment_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get deployment history for a model

        Args:
            model_id: Model identifier

        Returns:
            List of deployment records
        """
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT environment, endpoint, deployed_at, status
                FROM deployments
                WHERE model_id = ?
                ORDER BY deployed_at DESC
            """, (model_id,))

            deployments = []
            for row in cursor.fetchall():
                deployments.append({
                    "environment": row[0],
                    "endpoint": row[1],
                    "deployed_at": row[2],
                    "status": row[3]
                })

        return deployments

    def search_models(self, query: str) -> List[ModelMetadata]:
        """
        Search models by name or tags

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        query_lower = query.lower()

        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT metadata_json
                FROM models
                WHERE LOWER(name) LIKE ? OR LOWER(metadata_json) LIKE ?
                ORDER BY updated_at DESC
            """, (f"%{query_lower}%", f"%{query_lower}%"))

            models = []
            for result in cursor.fetchall():
                metadata = ModelMetadata.from_json(result[0])
                models.append(metadata)

        return models

    def export_registry(self, output_path: str) -> None:
        """
        Export registry to JSON file

        Args:
            output_path: Output file path
        """
        models = self.list_models()

        registry_data = {
            "version": "1.0.0",
            "exported_at": datetime.now().isoformat(),
            "model_count": len(models),
            "models": [model.to_dict() for model in models]
        }

        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)

        logger.info(f"Exported {len(models)} models to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics

        Returns:
            Statistics dictionary
        """
        with self._get_db() as conn:
            cursor = conn.cursor()

            # Total models
            cursor.execute("SELECT COUNT(*) FROM models")
            total_models = cursor.fetchone()[0]

            # Models by status
            cursor.execute("""
                SELECT deployment_status, COUNT(*)
                FROM models
                GROUP BY deployment_status
            """)
            status_counts = dict(cursor.fetchall())

            # Models by task
            cursor.execute("""
                SELECT task, COUNT(*)
                FROM models
                GROUP BY task
            """)
            task_counts = dict(cursor.fetchall())

            # Recent models
            cursor.execute("""
                SELECT name, version, updated_at
                FROM models
                ORDER BY updated_at DESC
                LIMIT 5
            """)
            recent_models = [
                {"name": row[0], "version": row[1], "updated_at": row[2]}
                for row in cursor.fetchall()
            ]

        return {
            "total_models": total_models,
            "models_by_status": status_counts,
            "models_by_task": task_counts,
            "recent_models": recent_models,
            "registry_path": str(self.registry_path),
            "database_path": self.db_path,
        }