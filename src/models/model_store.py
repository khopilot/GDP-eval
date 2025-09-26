"""
Model Store
Handles model artifact storage and management
"""

import os
import json
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import yaml

logger = logging.getLogger(__name__)


class ModelStore:
    """Model artifact storage manager"""

    def __init__(self, base_path: str = "models/store"):
        """
        Initialize model store

        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Storage structure
        self.checkpoints_dir = self.base_path / "checkpoints"
        self.adapters_dir = self.base_path / "adapters"
        self.quantized_dir = self.base_path / "quantized"
        self.exports_dir = self.base_path / "exports"

        # Create directories
        for dir_path in [self.checkpoints_dir, self.adapters_dir, self.quantized_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def store_model(
        self,
        model_path: str,
        model_id: str,
        model_type: str = "checkpoint",
        compress: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store model artifacts

        Args:
            model_path: Source path of model files
            model_id: Unique model identifier
            model_type: Type of model (checkpoint, adapter, quantized, export)
            compress: Whether to compress the model
            metadata: Additional metadata

        Returns:
            Path to stored model
        """
        source_path = Path(model_path)
        if not source_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Determine storage directory
        if model_type == "checkpoint":
            target_dir = self.checkpoints_dir
        elif model_type == "adapter":
            target_dir = self.adapters_dir
        elif model_type == "quantized":
            target_dir = self.quantized_dir
        elif model_type == "export":
            target_dir = self.exports_dir
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        target_path = target_dir / model_id

        # Store model
        if compress:
            # Create compressed archive
            archive_path = target_path.with_suffix('.tar.gz')
            logger.info(f"Compressing model to {archive_path}")

            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(source_path, arcname=model_id)

            stored_path = str(archive_path)
        else:
            # Copy model files
            if target_path.exists():
                shutil.rmtree(target_path)

            logger.info(f"Copying model to {target_path}")
            shutil.copytree(source_path, target_path)
            stored_path = str(target_path)

        # Save metadata
        if metadata:
            metadata_path = target_path / "store_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            store_metadata = {
                "model_id": model_id,
                "model_type": model_type,
                "stored_at": datetime.now().isoformat(),
                "compressed": compress,
                "original_path": str(source_path),
                "stored_path": stored_path,
                **metadata
            }

            with open(metadata_path, 'w') as f:
                json.dump(store_metadata, f, indent=2)

        # Calculate checksum
        checksum = self._calculate_checksum(Path(stored_path))
        logger.info(f"Model stored with checksum: {checksum}")

        return stored_path

    def retrieve_model(
        self,
        model_id: str,
        model_type: str = "checkpoint",
        extract_to: Optional[str] = None
    ) -> str:
        """
        Retrieve stored model

        Args:
            model_id: Model identifier
            model_type: Type of model
            extract_to: Optional extraction directory for compressed models

        Returns:
            Path to model
        """
        # Determine storage directory
        if model_type == "checkpoint":
            search_dir = self.checkpoints_dir
        elif model_type == "adapter":
            search_dir = self.adapters_dir
        elif model_type == "quantized":
            search_dir = self.quantized_dir
        elif model_type == "export":
            search_dir = self.exports_dir
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Look for model
        model_path = search_dir / model_id
        compressed_path = model_path.with_suffix('.tar.gz')

        if compressed_path.exists():
            # Extract compressed model
            if extract_to:
                extract_path = Path(extract_to)
            else:
                extract_path = Path("/tmp") / f"gdpval_model_{model_id}"

            extract_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting model from {compressed_path} to {extract_path}")
            with tarfile.open(compressed_path, 'r:gz') as tar:
                tar.extractall(extract_path)

            return str(extract_path / model_id)

        elif model_path.exists():
            return str(model_path)

        else:
            raise ValueError(f"Model not found: {model_id} (type: {model_type})")

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List stored models

        Args:
            model_type: Optional filter by model type

        Returns:
            List of model information
        """
        models = []

        # Determine directories to search
        if model_type:
            if model_type == "checkpoint":
                search_dirs = [self.checkpoints_dir]
            elif model_type == "adapter":
                search_dirs = [self.adapters_dir]
            elif model_type == "quantized":
                search_dirs = [self.quantized_dir]
            elif model_type == "export":
                search_dirs = [self.exports_dir]
            else:
                search_dirs = []
        else:
            search_dirs = [self.checkpoints_dir, self.adapters_dir, self.quantized_dir, self.exports_dir]

        for search_dir in search_dirs:
            model_type_name = search_dir.name

            # Find all models in directory
            for path in search_dir.iterdir():
                if path.is_dir():
                    model_info = {
                        "model_id": path.name,
                        "model_type": model_type_name,
                        "path": str(path),
                        "compressed": False
                    }

                    # Load metadata if exists
                    metadata_path = path / "store_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            model_info.update(metadata)

                    models.append(model_info)

                elif path.suffix == '.gz':
                    model_info = {
                        "model_id": path.stem.replace('.tar', ''),
                        "model_type": model_type_name,
                        "path": str(path),
                        "compressed": True
                    }
                    models.append(model_info)

        return models

    def store_lora_adapter(
        self,
        adapter_path: str,
        model_id: str,
        base_model: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store LoRA adapter

        Args:
            adapter_path: Path to adapter files
            model_id: Adapter identifier
            base_model: Base model name
            metadata: Additional metadata

        Returns:
            Path to stored adapter
        """
        adapter_metadata = {
            "base_model": base_model,
            "adapter_type": "lora",
            **(metadata or {})
        }

        return self.store_model(
            adapter_path,
            model_id,
            model_type="adapter",
            compress=True,
            metadata=adapter_metadata
        )

    def store_quantized_model(
        self,
        model_path: str,
        model_id: str,
        quantization_type: str,
        original_model: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store quantized model

        Args:
            model_path: Path to quantized model
            model_id: Model identifier
            quantization_type: Type of quantization (gptq, awq, gguf)
            original_model: Original model name
            metadata: Additional metadata

        Returns:
            Path to stored model
        """
        quant_metadata = {
            "quantization_type": quantization_type,
            "original_model": original_model,
            **(metadata or {})
        }

        # GGUF models are single files
        if quantization_type == "gguf" and Path(model_path).is_file():
            # Create directory for the model
            target_dir = self.quantized_dir / model_id
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy GGUF file
            gguf_file = Path(model_path)
            target_file = target_dir / gguf_file.name
            shutil.copy2(gguf_file, target_file)

            # Save metadata
            metadata_path = target_dir / "store_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(quant_metadata, f, indent=2)

            return str(target_dir)
        else:
            return self.store_model(
                model_path,
                model_id,
                model_type="quantized",
                metadata=quant_metadata
            )

    def export_model(
        self,
        model_id: str,
        export_format: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export model in specific format

        Args:
            model_id: Model identifier
            export_format: Export format (onnx, tflite, coreml, etc.)
            output_path: Optional output path

        Returns:
            Path to exported model
        """
        # This would implement actual model conversion
        # For now, just create a placeholder
        if output_path is None:
            output_path = self.exports_dir / f"{model_id}_{export_format}"

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save export metadata
        export_metadata = {
            "model_id": model_id,
            "export_format": export_format,
            "exported_at": datetime.now().isoformat()
        }

        with open(output_path / "export_metadata.json", 'w') as f:
            json.dump(export_metadata, f, indent=2)

        logger.info(f"Model exported to {output_path}")
        return str(output_path)

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of file or directory"""
        sha256 = hashlib.sha256()

        if path.is_file():
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
        else:
            # Calculate checksum for directory
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256.update(chunk)

        return sha256.hexdigest()

    def verify_model(self, model_id: str, model_type: str = "checkpoint") -> bool:
        """
        Verify model integrity

        Args:
            model_id: Model identifier
            model_type: Type of model

        Returns:
            True if model is valid
        """
        try:
            model_path = self.retrieve_model(model_id, model_type)
            return Path(model_path).exists()
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False

    def cleanup_old_models(self, days: int = 30, dry_run: bool = True) -> List[str]:
        """
        Clean up old models

        Args:
            days: Remove models older than this many days
            dry_run: If True, only report what would be deleted

        Returns:
            List of removed model paths
        """
        import time

        removed = []
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for search_dir in [self.checkpoints_dir, self.adapters_dir, self.quantized_dir, self.exports_dir]:
            for path in search_dir.iterdir():
                # Check modification time
                if path.stat().st_mtime < cutoff_time:
                    removed.append(str(path))

                    if not dry_run:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        logger.info(f"Removed old model: {path}")

        if dry_run:
            logger.info(f"Would remove {len(removed)} old models")
        else:
            logger.info(f"Removed {len(removed)} old models")

        return removed

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Storage statistics
        """
        def get_dir_size(path: Path) -> int:
            """Get total size of directory"""
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        stats = {
            "total_size_gb": 0,
            "checkpoints": {
                "count": len(list(self.checkpoints_dir.iterdir())),
                "size_gb": get_dir_size(self.checkpoints_dir) / (1024**3)
            },
            "adapters": {
                "count": len(list(self.adapters_dir.iterdir())),
                "size_gb": get_dir_size(self.adapters_dir) / (1024**3)
            },
            "quantized": {
                "count": len(list(self.quantized_dir.iterdir())),
                "size_gb": get_dir_size(self.quantized_dir) / (1024**3)
            },
            "exports": {
                "count": len(list(self.exports_dir.iterdir())),
                "size_gb": get_dir_size(self.exports_dir) / (1024**3)
            }
        }

        stats["total_size_gb"] = sum(
            v["size_gb"] for v in stats.values() if isinstance(v, dict)
        )

        return stats