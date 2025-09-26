"""
File Handlers for GDPval Framework
Handles multi-modal file processing for evaluation tasks
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MultiModalFileHandler:
    """Handles various file types used in evaluation tasks"""

    SUPPORTED_FORMATS = {
        '.json': 'json',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.txt': 'text',
        '.md': 'markdown',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.pdf': 'pdf'  # Placeholder for PDF support
    }

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize file handler

        Args:
            base_path: Base directory for file operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def read_file(self, file_path: Union[str, Path], **kwargs) -> Any:
        """
        Read file based on its extension

        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for specific file types

        Returns:
            File contents in appropriate format
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")

        file_type = self.SUPPORTED_FORMATS[extension]

        try:
            if file_type == 'json':
                return self._read_json(file_path)
            elif file_type == 'csv':
                return self._read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                return self._read_excel(file_path, **kwargs)
            elif file_type == 'text' or file_type == 'markdown':
                return self._read_text(file_path)
            elif file_type == 'yaml':
                return self._read_yaml(file_path)
            elif file_type == 'pdf':
                return self._read_pdf_placeholder(file_path)
            else:
                raise ValueError(f"Handler not implemented for {file_type}")

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def write_file(self, file_path: Union[str, Path], data: Any, **kwargs) -> None:
        """
        Write data to file based on extension

        Args:
            file_path: Path to write to
            data: Data to write
            **kwargs: Additional arguments for specific file types
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")

        file_type = self.SUPPORTED_FORMATS[extension]

        try:
            if file_type == 'json':
                self._write_json(file_path, data, **kwargs)
            elif file_type == 'csv':
                self._write_csv(file_path, data, **kwargs)
            elif file_type == 'excel':
                self._write_excel(file_path, data, **kwargs)
            elif file_type == 'text' or file_type == 'markdown':
                self._write_text(file_path, data)
            elif file_type == 'yaml':
                self._write_yaml(file_path, data)
            else:
                raise ValueError(f"Writer not implemented for {file_type}")

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            raise

    def _read_json(self, file_path: Path) -> Any:
        """Read JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: Any, indent: int = 2, **kwargs) -> None:
        """Write JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, **kwargs)

    def _read_csv(self, file_path: Path, as_dataframe: bool = False, **kwargs) -> Union[List[Dict], pd.DataFrame]:
        """Read CSV file"""
        if as_dataframe:
            return pd.read_csv(file_path, **kwargs)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)

    def _write_csv(self, file_path: Path, data: Union[List[Dict], pd.DataFrame], **kwargs) -> None:
        """Write CSV file"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False, **kwargs)
        else:
            if not data:
                return
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    def _read_excel(self, file_path: Path, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Read Excel file"""
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    def _write_excel(self, file_path: Path, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], **kwargs) -> None:
        """Write Excel file"""
        if isinstance(data, pd.DataFrame):
            data.to_excel(file_path, index=False, **kwargs)
        else:
            # Multiple sheets
            with pd.ExcelWriter(file_path, **kwargs) as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _read_text(self, file_path: Path) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _write_text(self, file_path: Path, data: str) -> None:
        """Write text file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)

    def _read_yaml(self, file_path: Path) -> Any:
        """Read YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _write_yaml(self, file_path: Path, data: Any) -> None:
        """Write YAML file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    def _read_pdf_placeholder(self, file_path: Path) -> Dict[str, Any]:
        """Placeholder for PDF reading (would require PyPDF2 or similar)"""
        logger.warning(f"PDF reading not fully implemented. Returning metadata for {file_path}")
        return {
            "file_path": str(file_path),
            "file_type": "pdf",
            "message": "PDF content extraction requires additional libraries (e.g., PyPDF2, pdfplumber)",
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0
        }

    def process_reference_files(self, reference_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a list of reference files from task definition

        Args:
            reference_files: List of reference file definitions

        Returns:
            Dictionary with processed file contents
        """
        processed = {}

        for ref_file in reference_files:
            file_id = ref_file.get('file_id', 'unknown')
            file_name = ref_file.get('file_name', '')

            if not file_name:
                logger.warning(f"No file name for reference {file_id}")
                continue

            file_path = self.base_path / 'data' / 'reference_files' / file_name

            try:
                # Check if file exists
                if not file_path.exists():
                    logger.warning(f"Reference file not found: {file_path}")
                    processed[file_id] = {
                        "status": "not_found",
                        "file_name": file_name
                    }
                    continue

                # Read file based on type
                file_type = ref_file.get('file_type', '')
                if file_type == 'spreadsheet' or file_name.endswith(('.xlsx', '.xls')):
                    content = self.read_file(file_path, as_dataframe=True)
                    processed[file_id] = {
                        "status": "loaded",
                        "file_name": file_name,
                        "type": "dataframe",
                        "shape": content.shape if isinstance(content, pd.DataFrame) else None,
                        "preview": content.head().to_dict() if isinstance(content, pd.DataFrame) else None
                    }
                elif file_type == 'csv' or file_name.endswith('.csv'):
                    content = self.read_file(file_path, as_dataframe=True)
                    processed[file_id] = {
                        "status": "loaded",
                        "file_name": file_name,
                        "type": "dataframe",
                        "shape": content.shape if isinstance(content, pd.DataFrame) else None
                    }
                elif file_type == 'json' or file_name.endswith('.json'):
                    content = self.read_file(file_path)
                    processed[file_id] = {
                        "status": "loaded",
                        "file_name": file_name,
                        "type": "json",
                        "content": content
                    }
                else:
                    content = self.read_file(file_path)
                    processed[file_id] = {
                        "status": "loaded",
                        "file_name": file_name,
                        "type": "text",
                        "content": str(content)[:1000]  # Truncate for preview
                    }

            except Exception as e:
                logger.error(f"Error processing reference file {file_name}: {e}")
                processed[file_id] = {
                    "status": "error",
                    "file_name": file_name,
                    "error": str(e)
                }

        return processed

    def save_evaluation_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        format: str = 'json'
    ) -> Path:
        """
        Save evaluation results to file

        Args:
            results: List of evaluation results
            output_dir: Output directory
            format: Output format (json, csv, excel)

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evaluation_results_{timestamp}"

        if format == 'json':
            file_path = output_dir / f"{filename}.json"
            self.write_file(file_path, results)
        elif format == 'csv':
            file_path = output_dir / f"{filename}.csv"
            df = pd.DataFrame(results)
            self.write_file(file_path, df)
        elif format == 'excel':
            file_path = output_dir / f"{filename}.xlsx"
            df = pd.DataFrame(results)
            self.write_file(file_path, df)
        else:
            raise ValueError(f"Unsupported output format: {format}")

        logger.info(f"Saved evaluation results to {file_path}")
        return file_path