# GDP-eval: Enterprise-Grade AI Model Evaluation Framework

**Multi-dimensional AI evaluation system with 451+ comprehensive tests**
*Specialized for Cambodia workforce development with Khmer language support*

## 🏆 What is GDP-eval?

GDP-eval is a production-ready evaluation framework designed to assess AI models' real-world impact on Cambodia's workforce. It provides enterprise-grade metrics through 5 evaluation dimensions (Capability, Safety, Robustness, Consistency, Behavioral) with 451+ comprehensive tests, focusing on professional task performance and economic value generation.

## 🎯 Key Features

### Core Capabilities
- **Enterprise Scoring System**: Multi-criteria evaluation (accuracy, completeness, technical correctness, language quality)
- **Differentiated Assessment**: No uniform scoring - each task evaluated on actual performance
- **Economic Impact Analysis**: Real-time calculation of productivity gains and cost savings
- **Professional Task Suite**: 5 Cambodia-specific professional scenarios across key sectors
- **Bilingual Excellence**: Native Khmer and English language support with code-switching detection

### Technical Excellence
- **Multi-Provider Support**: Grok, vLLM, Ollama, HuggingFace, and local models
- **Adaptive Timeouts**: Complexity-aware timeout management (30-240s based on task)
- **Enterprise Metrics**: 5-factor efficiency scoring (Success, Quality, Latency, Cost, Consistency)
- **Model Registry**: Centralized model versioning and performance tracking
- **High Performance**: vLLM integration for 10x faster inference

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for vLLM)
- 16GB+ RAM
- API keys for cloud providers (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/khopilot/GDP-eval.git
cd GDP-eval

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Run automated setup
bash setup.sh
```

## 📊 Testing Workflows

### 🚀 Enterprise Multi-Dimensional Evaluation (Recommended)

**Complete Enterprise Workflow:**
```bash
# Full enterprise-grade evaluation (5-15 minutes)
python run_full_workflow.py
```

This comprehensive workflow includes:
- ✅ Professional task evaluation (5 Cambodia-specific scenarios)
- ✅ Multi-dimensional enterprise assessment (5 evaluation modules)
- ✅ Safety and robustness testing
- ✅ Consistency and behavioral analysis
- ✅ Economic impact calculation
- ✅ Professional report generation (JSON, HTML, Text)

**Interactive Multi-Dimensional Testing:**
```bash
# Individual module testing with interactive menu
python test_multidimensional_evaluation.py

# Menu options:
# 1. 🏆 Comprehensive Evaluation (all 5 modules)
# 2. 🔬 Individual Module Testing
# 3. ⚖️ Suite Comparison (minimal vs comprehensive)
# 4. 🚀 Complete Testing Workflow
```

### 🧠 Evaluation Modules

| Module | Purpose | Sample Score | Status |
|--------|---------|-------------|---------|
| **Capability** | Reasoning, knowledge, creativity | 47.4/100 | ✅ |
| **Safety** | Toxicity, bias, hallucination detection | Variable | ✅ |
| **Robustness** | Adversarial attacks, edge cases | 77.7/100 | ✅ |
| **Consistency** | Temporal stability, cross-prompt coherence | 100.0/100 | ✅ |
| **Behavioral** | Helpfulness, harmlessness, honesty | 26.8/100 | ✅ |

### 📊 Legacy Professional Task Workflow

**Phase 1: Test with Baseline Model**
```bash
# Test with Grok API (or any baseline)
python test_grok_professional.py
```

**Phase 2: Compare with Fine-tuned Model**
```bash
# Edit compare_models.py to add your model
python compare_models.py
```

**Phase 3: Analyze Results**
```bash
# View detailed comparison
cat results/comparison/comparison_*.json
```

## 📊 Evaluation Datasets

### Test Distribution (451 Total Tests)

| Dataset | Tests | Categories | Description |
|---------|-------|------------|-------------|
| **capability_tests.json** | 65 | Logical, Technical, Knowledge | Reasoning and problem-solving |
| **safety_tests.json** | 82 | Toxicity, Bias, Misinformation | Content safety and ethics |
| **robustness_tests.json** | 74 | Edge cases, Recovery, Adversarial | Input handling and stability |
| **consistency_tests.json** | 90 | Temporal, Cross-prompt, Factual | Response coherence |
| **behavioral_tests.json** | 140 | Instructions, Context, Quality | User interaction patterns |

### Language Coverage
- **English**: 40% of tests
- **Khmer**: 30% of tests
- **Mixed/Bilingual**: 30% of tests

### Test Complexity Distribution
- **Simple (Level 1)**: 25% - Basic functionality
- **Moderate (Level 2)**: 50% - Standard use cases
- **Complex (Level 3)**: 25% - Advanced scenarios

## 🔬 Enterprise Testing Pipeline

### 1. **Environment Setup**
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install vLLM for high performance
pip install vllm

# Configure API keys in .env
GROK_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

### 2. **Model Configuration**
```yaml
# configs/models.yaml
models:
  baseline:
    - name: "grok-3"
      type: "grok"
      timeout: 120

  finetuned:
    - name: "your-model"
      type: "ollama"
      base_url: "http://localhost:11434"
```

### 3. **Professional Task Evaluation**
The framework tests 5 critical professional scenarios:

| Sector | Task | Complexity | Language | Success Metric |
|--------|------|------------|----------|----------------|
| Finance | SME Loan Assessment | 3 | Bilingual | Risk accuracy |
| Agriculture | Rice Farming Advisory | 3 | Khmer | Technical correctness |
| Tourism | Tour Package Design | 2 | English | Completeness |
| Manufacturing | Quality Control | 2 | English | Standards compliance |
| Healthcare | Patient Triage | 3 | Bilingual | Urgency assessment |

### 4. **Evaluation Criteria**
Each task is scored on 4 weighted criteria:
- **Accuracy** (30%): Correctness of information
- **Completeness** (30%): All requirements addressed
- **Technical Correctness** (20%): Domain-specific accuracy
- **Language Quality** (20%): Appropriate language use

### 5. **Metrics Collection**

#### Performance Metrics
- Success rate (task completion)
- Average latency (response time)
- Token usage and costs
- Throughput (tasks/second)

#### Quality Metrics
- Multi-criteria scores (0-100%)
- Pass/fail by criterion
- Average quality score
- Sector-specific performance

#### Economic Impact
- Time saved per day (hours)
- Annual cost savings (USD)
- Productivity gain (%)
- ROI calculation

### 6. **Enterprise Efficiency Score**
```
Efficiency = 0.35*Success + 0.25*Quality + 0.20*Latency + 0.10*Cost + 0.10*Consistency
```
- Adjusts for task complexity
- Considers quality-speed tradeoffs
- Tracks consistency across runs

## 📊 Enterprise Results Interpretation

### 🎯 Overall Score Ranges
- **Score > 75%**: Production-ready, enterprise deployment approved
- **Score 60-75%**: Suitable for pilot programs and staged rollout
- **Score 45-60%**: Development stage, significant improvements needed
- **Score < 45%**: Research phase, comprehensive retraining required

### 🚨 Risk Assessment Levels
- **LOW**: Ready for production deployment
- **MEDIUM**: Address identified issues before deployment
- **HIGH**: Significant concerns require resolution
- **CRITICAL**: Do not deploy - major safety/security issues

### 📋 Generated Reports
After running the enterprise workflow, check these locations:
- **JSON Reports**: `results/reports/json/` - Machine-readable for integration
- **HTML Reports**: `results/reports/html/` - Executive presentation format
- **Text Reports**: `results/reports/text/` - Detailed technical analysis

### ⏱️ Execution Metrics
- **Total Tests**: 60+ comprehensive evaluations per run
- **Duration**: 5-15 minutes for complete assessment
- **Modules**: 5 evaluation dimensions (Capability, Safety, Robustness, Consistency, Behavioral)
- **API Calls**: ~50 model inference calls for thorough testing

## 📈 Recent Improvements

### Version 2.1 - Complete Dataset Integration (Latest)
- ✅ **451+ Comprehensive Tests** - Fully integrated JSON datasets
- ✅ **Fixed All Critical Bugs**:
  - Capability module 'type' KeyError resolved
  - Consistency module 'prompt' field mapping fixed
  - Safety module 'max_risk' calculation added
  - Factual consistency 'fact_checks' bug fixed
- ✅ **BaseEvaluationModule** - Unified dataset loading across all modules
- ✅ **Enhanced Error Handling** - Prompt validation and sanitization
- ✅ **100% Module Success Rate** - All evaluation modules fully operational

### Version 2.0 - Enterprise Multi-Dimensional Edition
- ✅ **Enterprise Evaluation Suite** - 5-module comprehensive assessment
- ✅ **Fixed critical integration errors** - All modules now execute successfully
- ✅ **Fixed uniform 60% scoring** - Now provides differentiated scores
- ✅ **Fixed sector mappings** - Resolved finance_banking → finance issues
- ✅ **Removed hardcoded values** - Real economic calculations
- ✅ **Fixed NumPy deprecation** - Custom IRR implementation
- ✅ **Enhanced metrics** - Enterprise-grade 5-factor scoring
- ✅ **Professional reports** - Multi-format output (JSON, HTML, Text)

### Performance Benchmarks
| Metric | V1.0 | V2.0 | Improvement |
|--------|------|------|-------------|
| Total Tests | 5 tasks | 64+ evaluations | +1,180% |
| Evaluation Modules | 1 | 5 comprehensive | +400% |
| Efficiency Score | 40.8% | 95.5% | +134% |
| Score Variance | 0% (uniform) | Realistic differentiation | Real assessment |
| System Reliability | 60% | 100% | +67% |
| Report Formats | 1 | 3 (JSON/HTML/Text) | +200% |

## 🇰🇭 Khmer Language Excellence

### Advanced Tokenization
- Character-level with grapheme clustering
- Syllable segmentation (ZWSP-aware)
- Word boundary detection
- Code-switching detection (Khmer/English)
- Bilingual context handling

### Khmer-Specific Metrics
- **Khmer BLEU**: Multi-level evaluation
- **Character Accuracy**: Precise matching
- **Syllable F1**: Segmentation quality
- **Language Distribution**: Khmer/English ratio
- **Script Detection**: Automatic language identification

## 📁 Project Structure
```
GDP-eval/
├── configs/                 # Model and system configuration
│   ├── models.yaml         # Model registry
│   ├── vllm_config.yaml    # vLLM settings
│   └── deployment.yaml     # Environment configs
├── src/
│   ├── providers/          # LLM provider implementations
│   │   ├── grok_provider.py
│   │   ├── enterprise_grok_provider.py
│   │   ├── vllm_provider.py
│   │   └── ollama_provider.py
│   ├── metrics/            # Evaluation metrics
│   │   ├── khmer_metrics.py
│   │   └── performance_metrics.py
│   ├── tasks/              # Task management
│   │   ├── task_converter.py
│   │   └── professional_task_manager.py
│   ├── analysis/           # Economic analysis
│   │   ├── gdp_analyzer.py
│   │   └── impact_calculator.py
│   └── core/               # Core framework
├── test_grok_professional.py  # Main test script
├── compare_models.py          # Model comparison tool
└── results/                   # Evaluation outputs
```

## 🔧 Advanced Usage

### Custom Provider Integration
```python
from src.providers.base_provider import BaseLLMProvider

class YourProvider(BaseLLMProvider):
    async def generate(self, prompt, **kwargs):
        # Your implementation
        return LLMResponse(...)
```

### Adding Professional Tasks
```python
from src.tasks.task_converter import TaskConverter

converter = TaskConverter()
task = ProfessionalTask(
    task_id="custom_001",
    title="Your Task",
    sector="your_sector",
    complexity=3,
    evaluation_criteria=[
        {"criterion_id": "accuracy", "weight": 0.3},
        {"criterion_id": "completeness", "weight": 0.3}
    ]
)
```

### Fine-tuning Workflow
```bash
# 1. Prepare your model
ollama create your-model -f Modelfile

# 2. Update compare_models.py
model_configs = [
    {"name": "grok-baseline", "type": "grok", ...},
    {"name": "your-model", "type": "ollama", ...}
]

# 3. Run comparison
python compare_models.py

# 4. Analyze improvements
cat results/comparison/comparison_*.json
```

## 📊 Interpreting Results

### Success Indicators
- **Score > 70%**: Production-ready
- **Score 60-70%**: Suitable for pilot programs
- **Score < 60%**: Needs improvement

### Key Metrics
- **Efficiency > 90%**: Excellent performance
- **Latency < 30s**: Acceptable for professional use
- **Cost < $0.02/task**: Economically viable

## 👤 Author

**Nicolas Delrieu** - AI Consultant
📱 Phone: +855 92 332 554
🔧 Specializing in multilingual NLP and economic AI applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📝 License

This project is proprietary software. All rights reserved.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/khopilot/GDP-eval/issues)
- **Documentation**: See `docs/` directory
- **Contact**: +855 92 332 554

---

**GDP-eval**: *Measuring AI's real impact on Cambodia's digital economy*
