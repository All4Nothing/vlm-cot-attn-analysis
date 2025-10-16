# VLM Chain-of-Thought Attention Analysis

A comprehensive framework for analyzing attention patterns in Vision-Language Models, with a focus on chain-of-thought reasoning.

## 📁 Project Structure

```
vlm-cot-attn-analysis/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Image data for experiments
├── models/                      # Model implementations
│   ├── llava/                   # LLaVA model package
│   │   ├── config.py           # Model configuration
│   │   ├── model_loader.py     # Model loading utilities
│   │   ├── inference.py # Core inference engine
│   │   ├── image_processor.py  # Image preprocessing
│   │   ├── main.py            # CLI interface
│   │   ├── example_usage.py   # Basic usage examples
│   │   └── model_weights/     # Downloaded model weights
│   └── utils/                  # Model utilities
└── src/                        # Analysis and research code
    ├── analyzers/              # Attention analysis tools
    │   ├── attention_analyzer.py  # Core attention analysis
    │   └── llava_analyzer.py      # Integrated LLaVA analyzer
    ├── examples/               # Usage examples
    │   ├── basic_usage.py      # Basic inference examples
    │   └── attention_analysis.py # Advanced analysis examples
    ├── experiments/            # Research experiments
    └── utils/                  # Analysis utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd vlm-cot-attn-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
# Basic LLaVA inference
from models.llava import LLaVAInferenceEngine

engine = LLaVAInferenceEngine()
engine.load_model()
result = engine.generate_response("path/to/image.jpg", "What do you see?")
print(result['response'])
```

### 3. Advanced Attention Analysis

```python
# Comprehensive attention analysis
from src.analyzers import LLaVAAnalyzer

analyzer = LLaVAAnalyzer()
analyzer.load_model()
result = analyzer.analyze_image_response(
    image="path/to/image.jpg",
    question="What do you see?",
    analyze_attention=True
)

# Access attention analysis results
attention_data = result['attention_analysis']
print(f"Image attention summary: {attention_data['image_attention']['summary']}")
```

## 📚 Usage Examples

### Run Basic Examples
```bash
# Basic inference
python src/examples/basic_usage.py

# Advanced attention analysis
python src/examples/attention_analysis.py
```

### Programmatic Usage
```python
# Run examples programmatically
from src.examples import run_basic_example, run_attention_example

run_basic_example()      # Basic inference
run_attention_example()  # Attention analysis
```

## 🧠 Analysis Capabilities

### Attention Analysis Features
- **Image-Text Attention Mapping**: Analyze how the model attends to image regions
- **Head Specialization Analysis**: Study attention head specialization patterns
- **Statistical Analysis**: Compute entropy, variance, and other attention statistics
- **Comparative Analysis**: Compare attention patterns across different questions
- **Batch Processing**: Analyze multiple image-question pairs efficiently

### Supported Models
- **LLaVA-1.6-Vicuna-7B**: Primary focus with comprehensive support
- **Extensible**: Framework designed for easy addition of other VLMs

## 🔧 Configuration

### GPU Configuration
```python
from models.llava import Config

# Set specific GPU
Config.set_cuda_device("0")

# Or use convenience functions
from src.analyzers import create_llava_analyzer
analyzer = create_llava_analyzer(gpu_device="0")
```

### Model Configuration
```python
# Custom generation settings
config = Config()
config.GENERATION_CONFIG["max_new_tokens"] = 256
config.GENERATION_CONFIG["temperature"] = 0.1

# Use with inference engine
engine = LLaVAInferenceEngine(config)
```

## 📊 Research Applications

This framework is designed for:
- **Attention Pattern Studies**: Understanding how VLMs process visual information
- **Chain-of-Thought Analysis**: Analyzing reasoning patterns in VLM responses
- **Model Interpretability**: Making VLM decision processes more transparent
- **Comparative Studies**: Comparing different models or configurations

## 🛠️ Development

### Adding New Analyzers
```python
# Create new analyzer in src/analyzers/
class CustomAnalyzer:
    def __init__(self, config=None):
        # Initialize analyzer
        pass
    
    def analyze_custom_pattern(self, attention_data):
        # Implement custom analysis
        pass
```

### Adding New Experiments
```python
# Create experiment in src/experiments/
def experiment_attention_evolution():
    # Implement experiment logic
    pass
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (recommended)
- 16GB+ GPU memory for LLaVA-7B

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your analysis tools to `src/analyzers/` or experiments to `src/experiments/`
4. Update documentation
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- LLaVA team for the original model
- Hugging Face for the transformers library
- Research community for attention analysis methodologies