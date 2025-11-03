# AI Tutor Response Evaluation - Mistake Identification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/transformers/)

A machine learning solution for evaluating AI tutor responses by identifying whether they correctly address student mistakes. Built for the IndoML Datathon challenge.

## 👥 Contributors

- [@Kweenbee187](https://github.com/Kweenbee187)
- [@tituatgithub](https://github.com/tituatgithub)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🎯 Overview

This project tackles the challenge of automatically evaluating AI tutor responses by classifying whether they correctly identify and address student mistakes. The system uses state-of-the-art transformer models with stratified k-fold cross-validation to achieve robust classification across three categories:

- **Yes** - The tutor correctly identifies the mistake
- **No** - The tutor fails to identify the mistake
- **To some extent** - The tutor partially identifies the mistake

## ✨ Features

- **Advanced NLP Model**: Utilizes Microsoft's DeBERTa-v3-base transformer
- **Robust Training**: 5-fold stratified cross-validation for reliable performance
- **Class Imbalance Handling**: 
  - Minimal but effective data augmentation for minority class
  - Weighted cross-entropy loss
  - Threshold optimization
- **Comprehensive Evaluation**: Per-fold and cross-fold metrics with detailed per-class analysis
- **Ensemble Predictions**: Aggregates predictions from all folds for improved accuracy

## 📊 Dataset

The dataset is from the [UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation) repository.

### Data Structure

```
Training Set: 2,476 samples
├── Yes: 1,932 samples (78.0%)
├── No: 370 samples (15.0%)
└── To some extent: 174 samples (7.0%)

Test Set: 1,214 samples
```

### Input Format

```json
{
  "conversation_id": "unique_id",
  "conversation_history": "Student: ...\nTutor: ...",
  "tutor_responses": {
    "tutor_name": {
      "response": "tutor response text",
      "annotation": {
        "Mistake_Identification": "Yes/No/To some extent"
      }
    }
  }
}
```

## 🏗️ Model Architecture

### Base Model
- **Model**: `microsoft/deberta-v3-base`
- **Architecture**: DeBERTa v3 (Decoding-enhanced BERT with disentangled attention)
- **Parameters**: ~184M
- **Max Sequence Length**: 384 tokens

### Training Strategy

1. **Text Preprocessing**
   - Concatenates conversation history (last 3 turns) with tutor response
   - Uses `[SEP]` tokens as delimiters
   - Truncates to 384 tokens maximum

2. **Data Augmentation**
   - Simple augmentation for minority class ("To some extent")
   - Adds linguistic variations without over-complicating

3. **Class Weighting**
   - Computed using sklearn's balanced class weights
   - Additional 1.5x boost for minority class
   - Clipped to [0.8, 3.0] range

4. **Training Configuration**
   ```python
   Learning Rate: 3e-5
   Batch Size: 16
   Epochs: 3
   Weight Decay: 0.01
   Warmup Steps: 100
   Early Stopping: 2 epochs patience
   ```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone this repository
git clone [https://github.com/tituatgithub/INDO_ML.git]
cd ai-tutor-evaluation

# Install required packages
pip install -q transformers datasets accelerate scikit-learn tqdm sentence-transformers torch

# Clone the dataset repository
git clone https://github.com/kaushal0494/UnifyingAITutorEvaluation.git
```

## 💻 Usage

### Training the Model

```bash
# Navigate to the project directory
cd ai-tutor-evaluation

# Run the training script
python train_model.py
```

### Making Predictions

```python
from inference import load_model, predict

# Load trained model
model = load_model('path/to/model')

# Make predictions
predictions = predict(model, test_data)
```

### Quick Start with Notebook

Open and run `AI_Tutor_Evaluation.ipynb` for a complete walkthrough including:
- Data loading and exploration
- Model training
- Evaluation metrics
- Prediction generation

## 📈 Results

### Official Test Set Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **87.48%** |
| **Test Macro F1** | **69.94%** |

### Cross-Fold Validation Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.39% |
| **Macro F1** | 81.62% |
| **Macro Precision** | 82.98% |
| **Macro Recall** | 80.60% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **No** | 75.26% ± 9.49% | 72.43% ± 5.03% | 73.62% ± 6.41% | 370 |
| **To some extent** | 82.72% ± 2.58% | 76.05% ± 3.56% | 79.13% ± 1.08% | 522* |
| **Yes** | 90.95% ± 1.19% | 93.32% ± 1.71% | 92.10% ± 0.59% | 1,932 |

*After augmentation

### Training Progress

The model shows consistent improvement across folds with strong validation performance:

- **Fold 1**: F1 = 79.48%
- **Fold 2**: F1 = 84.91%
- **Fold 3**: F1 = 78.41%
- **Fold 4**: F1 = 83.66%
- **Fold 5**: F1 = 81.63%

**Final test set evaluation:**
- **Macro F1 Score: 69.94%**
- **Accuracy: 87.48%**

The difference between validation (81.62%) and test (69.94%) F1 scores suggests the minority class ("To some extent") is more challenging in the test set, which is common in imbalanced classification tasks.

### Threshold Analysis

Optimal threshold for "To some extent" class: **0.45-0.50**

## 📁 Project Structure

```
ai-tutor-evaluation/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── notebooks/
│   └── AI_Tutor_Evaluation.ipynb     # Main training notebook
│
├── src/
│   ├── __init__.py
│   ├── train_model.py                # Training script
│   ├── data_processing.py            # Data loading and preprocessing
│   ├── model.py                      # Model architecture and training
│   ├── inference.py                  # Prediction generation
│   └── utils.py                      # Utility functions
│
├── config/
│   └── config.yaml                   # Configuration parameters
│
├── data/
│   ├── README.md                     # Data documentation
│   └── .gitkeep                      # Keep folder in git
│
├── models/
│   └── .gitkeep                      # Saved models (not tracked)
│
├── outputs/
│   ├── predictions.json              # Generated predictions
│   └── metrics/                      # Evaluation metrics
│
└── docs/
    ├── model_architecture.md         # Detailed model documentation
    └── results_analysis.md           # Results analysis
```

## 🔧 Configuration

Key parameters can be adjusted in `config/config.yaml`:

```yaml
model:
  name: "microsoft/deberta-v3-base"
  max_length: 384
  
training:
  learning_rate: 3e-5
  batch_size: 16
  num_epochs: 3
  n_folds: 5
  seed: 42
  
augmentation:
  enabled: true
  minority_class_multiplier: 2
  
class_weights:
  minority_boost: 1.5
  clip_range: [0.8, 3.0]
```

## 🛠️ Technical Details

### Dependencies

```
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
sentence-transformers>=2.2.0
tqdm>=4.65.0
```

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU only (slow training)
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Training Time**: ~35 minutes for 5-fold CV on T4 GPU

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{ai-tutor-evaluation-2025,
  title={AI Tutor Response Evaluation: Mistake Identification},
  author={Kweenbee187 and tituatgithub},
  year={2025},
  publisher={GitHub},
  url={https://github.com/tituatgithub/INDO_ML}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [UnifyingAITutorEvaluation](https://github.com/kaushal0494/UnifyingAITutorEvaluation) for providing the dataset
- IndoML Datathon organizers
- Hugging Face for the Transformers library
- Microsoft Research for the DeBERTa model

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact:
- [@Kweenbee187](https://github.com/Kweenbee187)
- [@tituatgithub](https://github.com/tituatgithub)

---

**Note**: This project was developed as part of the IndoML Datathon challenge. The dataset and problem statement are provided by the competition organizers.
