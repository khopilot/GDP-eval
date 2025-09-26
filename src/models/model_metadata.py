"""
Model Metadata Schemas
Comprehensive metadata tracking for fine-tuned models
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json


class ModelTask(Enum):
    """Supported model tasks"""
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    FEATURE_EXTRACTION = "feature-extraction"


class ModelFramework(Enum):
    """Model frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    GGUF = "gguf"


class QuantizationType(Enum):
    """Quantization methods"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q5_K_M = "gguf_q5_k_m"
    GGUF_Q8_0 = "gguf_q8_0"


class DeploymentStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class DatasetInfo:
    """Information about training dataset"""
    name: str
    version: Optional[str] = None
    size: Optional[int] = None  # Number of samples
    languages: List[str] = field(default_factory=list)
    task: Optional[str] = None
    source: Optional[str] = None
    license: Optional[str] = None
    description: Optional[str] = None
    khmer_percentage: Optional[float] = None  # Percentage of Khmer content
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration details"""
    method: str  # fine-tuning, LoRA, QLoRA, etc.
    base_model: str
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    mixed_precision: Optional[str] = None  # fp16, bf16, fp32
    optimizer: Optional[str] = None
    scheduler: Optional[str] = None

    # LoRA specific
    lora_rank: Optional[int] = None
    lora_alpha: Optional[float] = None
    lora_dropout: Optional[float] = None
    target_modules: Optional[List[str]] = None

    # Hardware
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    training_time_hours: Optional[float] = None

    # Additional configs
    seed: Optional[int] = None
    max_sequence_length: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    use_flash_attention: Optional[bool] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    # General metrics
    perplexity: Optional[float] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None

    # Generation metrics
    bleu_score: Optional[float] = None
    rouge_l: Optional[float] = None
    meteor: Optional[float] = None

    # Khmer-specific metrics
    khmer_bleu: Optional[float] = None
    khmer_character_accuracy: Optional[float] = None
    khmer_word_accuracy: Optional[float] = None
    code_switching_accuracy: Optional[float] = None

    # Inference performance
    latency_ms_p50: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    latency_ms_p99: Optional[float] = None
    throughput_tokens_per_second: Optional[float] = None
    memory_usage_gb: Optional[float] = None

    # Task-specific metrics
    task_metrics: Dict[str, float] = field(default_factory=dict)

    # Evaluation dataset
    eval_dataset: Optional[str] = None
    eval_samples: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareRequirements:
    """Hardware requirements for model"""
    min_gpu_memory_gb: Optional[float] = None
    recommended_gpu_memory_gb: Optional[float] = None
    min_ram_gb: Optional[float] = None
    recommended_ram_gb: Optional[float] = None
    gpu_type: Optional[str] = None  # e.g., "NVIDIA A100", "RTX 3090"
    cpu_cores: Optional[int] = None
    supports_cpu_inference: bool = True
    supports_mps: bool = False  # Apple Metal Performance Shaders
    requires_flash_attention: bool = False
    optimal_batch_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KhmerCapabilities:
    """Khmer language specific capabilities"""
    supports_khmer: bool = True
    khmer_tokenizer: Optional[str] = None
    khmer_vocabulary_size: Optional[int] = None
    handles_zero_width_spaces: bool = False
    handles_khmer_numerals: bool = False
    handles_khmer_dates: bool = False
    supports_code_switching: bool = False
    dialect_support: List[str] = field(default_factory=list)  # e.g., ["central", "northern"]
    script_variants: List[str] = field(default_factory=list)  # e.g., ["modern", "classical"]

    # Domain expertise
    specialized_domains: List[str] = field(default_factory=list)  # e.g., ["finance", "legal", "medical"]

    # Quality metrics
    khmer_fluency_score: Optional[float] = None
    cultural_appropriateness_score: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Complete metadata for a model"""
    # Basic info (required fields first)
    model_id: str
    name: str
    version: str
    task: ModelTask

    # Optional fields with defaults
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Model details
    framework: ModelFramework = ModelFramework.PYTORCH
    model_size_gb: Optional[float] = None
    num_parameters: Optional[int] = None
    quantization: QuantizationType = QuantizationType.NONE

    # Training information
    training_config: Optional[TrainingConfig] = None
    dataset_info: Optional[DatasetInfo] = None

    # Performance
    performance_metrics: Optional[PerformanceMetrics] = None
    hardware_requirements: Optional[HardwareRequirements] = None

    # Khmer capabilities
    khmer_capabilities: Optional[KhmerCapabilities] = None

    # Languages and domains
    languages: List[str] = field(default_factory=lambda: ["km", "en"])
    domains: List[str] = field(default_factory=list)

    # Deployment
    deployment_status: DeploymentStatus = DeploymentStatus.DEVELOPMENT
    serving_framework: Optional[str] = None  # vllm, tgi, triton
    endpoints: List[str] = field(default_factory=list)

    # Documentation
    description: Optional[str] = None
    use_cases: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: Optional[str] = None

    # Versioning and lineage
    base_model: Optional[str] = None
    parent_version: Optional[str] = None
    derived_models: List[str] = field(default_factory=list)

    # Legal
    license: Optional[str] = None
    citation: Optional[str] = None

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)

        # Convert enums to strings
        if self.task:
            data['task'] = self.task.value
        if self.framework:
            data['framework'] = self.framework.value
        if self.quantization:
            data['quantization'] = self.quantization.value
        if self.deployment_status:
            data['deployment_status'] = self.deployment_status.value

        # Convert datetime to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary"""
        # Convert string enums back
        if 'task' in data:
            data['task'] = ModelTask(data['task'])
        if 'framework' in data:
            data['framework'] = ModelFramework(data['framework'])
        if 'quantization' in data:
            data['quantization'] = QuantizationType(data['quantization'])
        if 'deployment_status' in data:
            data['deployment_status'] = DeploymentStatus(data['deployment_status'])

        # Convert ISO strings to datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        # Convert nested dataclasses
        if 'training_config' in data and data['training_config']:
            data['training_config'] = TrainingConfig(**data['training_config'])
        if 'dataset_info' in data and data['dataset_info']:
            data['dataset_info'] = DatasetInfo(**data['dataset_info'])
        if 'performance_metrics' in data and data['performance_metrics']:
            data['performance_metrics'] = PerformanceMetrics(**data['performance_metrics'])
        if 'hardware_requirements' in data and data['hardware_requirements']:
            data['hardware_requirements'] = HardwareRequirements(**data['hardware_requirements'])
        if 'khmer_capabilities' in data and data['khmer_capabilities']:
            data['khmer_capabilities'] = KhmerCapabilities(**data['khmer_capabilities'])

        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics"""
        if not self.performance_metrics:
            self.performance_metrics = PerformanceMetrics()

        for key, value in metrics.items():
            if hasattr(self.performance_metrics, key):
                setattr(self.performance_metrics, key, value)
            else:
                self.performance_metrics.task_metrics[key] = value

        self.updated_at = datetime.now()

    def add_tag(self, tag: str) -> None:
        """Add a tag"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()

    def set_deployment_status(self, status: DeploymentStatus) -> None:
        """Update deployment status"""
        self.deployment_status = status
        self.updated_at = datetime.now()

    def add_endpoint(self, endpoint: str) -> None:
        """Add serving endpoint"""
        if endpoint not in self.endpoints:
            self.endpoints.append(endpoint)
            self.updated_at = datetime.now()

    def is_production_ready(self) -> bool:
        """Check if model is production ready"""
        return (
            self.deployment_status in [DeploymentStatus.STAGING, DeploymentStatus.PRODUCTION]
            and self.performance_metrics is not None
            and self.hardware_requirements is not None
        )

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Model: {self.name} (v{self.version})",
            f"ID: {self.model_id}",
            f"Task: {self.task.value if self.task else 'Unknown'}",
            f"Status: {self.deployment_status.value if self.deployment_status else 'Unknown'}",
        ]

        if self.performance_metrics:
            if self.performance_metrics.khmer_bleu:
                lines.append(f"Khmer BLEU: {self.performance_metrics.khmer_bleu:.3f}")
            if self.performance_metrics.latency_ms_p50:
                lines.append(f"Latency (P50): {self.performance_metrics.latency_ms_p50:.1f}ms")

        if self.hardware_requirements:
            lines.append(f"GPU Memory: {self.hardware_requirements.min_gpu_memory_gb}GB")

        if self.khmer_capabilities and self.khmer_capabilities.supports_khmer:
            lines.append("✓ Khmer language support")

        return "\n".join(lines)