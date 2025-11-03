# INDO_ML - Complete Project Structure

**Repository**: https://github.com/Kweenbee187/INDO_ML_2025  
**Contributors**: @Kweenbee187 & @tituatgithub  
**Test Results**: Accuracy 87.48% | Macro F1 69.94%

---

## 📁 Directory Structure

```
INDO_ML/
│
├── README.md                          # Main project documentation
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── SETUP.md                          # Installation & setup guide
├── CONTRIBUTING.md                    # Contribution guidelines
├── PROJECT_STRUCTURE.md              # This file
│
├── config/
│   └── config.yaml                   # Configuration parameters
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── train_model.py                # Main training script ⭐
│   ├── model.py                      # Model architecture & trainer
│   ├── data_processing.py            # Data loading & preprocessing
│   └── utils.py                      # Utility functions
│
├── notebooks/
│   └── AI_Tutor_Evaluation.ipynb    # Jupyter notebook walkthrough
│
├── data/                             # Data directory
│   ├── README.md                     # Data documentation
│   └── .gitkeep
│
├── models/                           # Saved model checkpoints
│   └── .gitkeep                      # (generated during training)
│
├── outputs/                          # Results & predictions
│   ├── predictions/                  # Prediction JSON files
│   ├── metrics/                      # Evaluation metrics
│   └── logs/                         # Training logs
│
└── docs/                             # Additional documentation
    ├── model_architecture.md         # Detailed architecture docs
    └── results_analysis.md           # Results analysis
```

---

## 📄 File Descriptions

### Root Files

| File | Description | Size | Essential |
|------|-------------|------|-----------|
| `README.md` | Main project documentation with setup instructions | ~15KB | ✅ Yes |
| `LICENSE` | MIT License for the project | ~1KB | ✅ Yes |
| `requirements.txt` | Python package dependencies | ~500B | ✅ Yes |
| `.gitignore` | Files to exclude from version control | ~1KB | ✅ Yes |
| `SETUP.md` | Detailed setup and troubleshooting guide | ~8KB | ⭐ Recommended |
| `CONTRIBUTING.md` | Guidelines for contributors | ~6KB | 📝 Optional |
| `PROJECT_STRUCTURE.md` | This file - project structure overview | ~4KB | 📝 Optional |

### Configuration (`config/`)

| File | Description | Purpose |
|------|-------------|---------|
| `config.yaml` | Hyperparameters and model settings | Centralized configuration management |

**Key parameters in config.yaml:**
```yaml
model:
  name: "microsoft/deberta-v3-base"
  max_length: 384

training:
  learning_rate: 3e-5
  batch_size: 16
  num_epochs: 3
  n_folds: 5
```

### Source Code (`src/`)

| File | Lines | Description | Main Functions |
|------|-------|-------------|----------------|
| `train_model.py` | ~400 | Main training pipeline | `train_model()`, `main()` |
| `model.py` | ~350 | Model architecture & trainer | `ResponseDataset`, `WeightedCrossEntropyTrainer`, `compute_metrics()` |
| `data_processing.py` | ~300 | Data loading & preprocessing | `load_training_data()`, `minimal_augment()`, `concat_text()` |
| `utils.py` | ~250 | Utility functions | `set_seed()`, `check_gpu()`, `print_metrics_summary()` |

#### `train_model.py` - Main Training Script

**Purpose**: Complete training pipeline with k-fold cross-validation

**Key Features**:
- 5-fold stratified cross-validation
- Data augmentation for minority class
- Ensemble predictions from all folds
- Comprehensive metrics tracking

**Usage**:
```bash
python src/train_model.py
```

**Output**:
- Trained models for each fold
- Predictions JSON file
- Training logs and metrics

#### `model.py` - Model Architecture

**Purpose**: Define model components and training utilities

**Key Classes**:
- `ResponseDataset`: PyTorch dataset for text data
- `WeightedCrossEntropyTrainer`: Custom trainer with weighted loss
- Metric computation functions

**Example**:
```python
from src.model import ResponseDataset, WeightedCrossEntropyTrainer

dataset = ResponseDataset(texts, labels, tokenizer, max_length=384)
trainer = WeightedCrossEntropyTrainer(model=model, class_weights=weights)
```

#### `data_processing.py` - Data Utilities

**Purpose**: Handle all data loading and preprocessing

**Key Functions**:
- `load_training_data()`: Load and flatten JSON training data
- `load_test_data()`: Load test data
- `minimal_augment()`: Augment minority class samples
- `concat_text()`: Combine conversation history with response
- `encode_labels()`: Convert text labels to integers

**Example**:
```python
from src.data_processing import load_training_data, minimal_augment

df = load_training_data("data/trainset.json")
df_aug = minimal_augment(df, multiplier=2)
```

#### `utils.py` - Utility Functions

**Purpose**: Helper functions for common tasks

**Key Functions**:
- `set_seed()`: Set random seeds for reproducibility
- `check_gpu()`: Check CUDA availability
- `print_metrics_summary()`: Format and print metrics
- `create_output_directory()`: Create directory structure
- `format_time()`: Format seconds to readable time

**Example**:
```python
from src.utils import set_seed, check_gpu, print_metrics_summary

set_seed(42)
check_gpu()
print_metrics_summary(metrics, title="Results")
```

---

## 🔄 Data Flow

```
1. DATA LOADING
   trainset.json → load_training_data() → DataFrame

2. DATA PREPROCESSING
   DataFrame → minimal_augment() → Augmented DataFrame
   → encode_labels() → Encoded labels
   → concat_text() → Preprocessed texts

3. MODEL TRAINING
   texts + labels → ResponseDataset → DataLoader
   → WeightedCrossEntropyTrainer → Trained Model

4. PREDICTION
   test data → Model ensemble → Predictions
   → create_prediction_json() → predictions.json

5. EVALUATION
   predictions + true labels → compute_metrics()
   → Results summary
```

---

## 🚀 Quick Start by File

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
**Uses**: `requirements.txt`

### Step 2: Prepare Data
```bash
git clone https://github.com/kaushal0494/UnifyingAITutorEvaluation.git
```
**Output**: Data in `UnifyingAITutorEvaluation/IndoML_Datathon/data/`

### Step 3: Configure Training
**Edit**: `config/config.yaml` (optional)

**Customize**:
- Model name
- Learning rate
- Batch size
- Number of folds

### Step 4: Run Training
```bash
python src/train_model.py
```
**Uses**: 
- `src/train_model.py` (main)
- `src/model.py` (architecture)
- `src/data_processing.py` (data)
- `src/utils.py` (utilities)

**Output**:
- `outputs/predictions/predictions.json`
- `outputs/metrics/*.json`
- `models/fold_*` (model checkpoints)

### Step 5: View Results
Check console output or:
```bash
cat outputs/metrics/summary.json
```

---

## 📝 File Dependencies

### `train_model.py` depends on:
- ✅ `model.py` - Model classes
- ✅ `data_processing.py` - Data loading
- ✅ `utils.py` - Utilities
- ✅ `config/config.yaml` - Configuration (optional)

### `model.py` depends on:
- ✅ `transformers` - Hugging Face models
- ✅ `torch` - PyTorch
- ✅ `sklearn` - Metrics

### `data_processing.py` depends on:
- ✅ `pandas` - DataFrame operations
- ✅ `numpy` - Array operations
- ✅ `sklearn` - Label encoding

### `utils.py` depends on:
- ✅ `torch` - Device management
- ✅ Standard library only

---

## 🔧 Customization Guide

### Change Model Architecture

**File**: `config/config.yaml`
```yaml
model:
  name: "roberta-base"  # Change this
  max_length: 512       # Adjust if needed
```

### Modify Training Parameters

**File**: `config/config.yaml`
```yaml
training:
  learning_rate: 2e-5   # Lower for stability
  batch_size: 8         # Reduce if OOM
  num_epochs: 5         # More epochs
```

### Add Custom Augmentation

**File**: `src/data_processing.py`

Add new augmentation in `minimal_augment()` function:
```python
if multiplier >= 3:
    aug3 = row.copy()
    aug3['response'] = "Perhaps, " + row['response']
    rows.append(aug3.to_dict())
```

### Change Evaluation Metrics

**File**: `src/model.py`

Modify `compute_metrics()` function:
```python
def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    
    return {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "
