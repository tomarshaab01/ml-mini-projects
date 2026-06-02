# ML Mini Projects — Setup Guide

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| pip | 23+ |
| Git | any |
| CUDA (optional) | 11.8+ for GPU training |

---

## 1. Clone & Install

```bash
git clone https://github.com/tomarshaab01/ml-mini-projects.git
cd ml-mini-projects

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Run Notebooks

```bash
jupyter notebook
```

Open notebooks in order:

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_data_preprocessing.ipynb` | Imputation, encoding, scaling |
| 02 | `02_classification_iris.ipynb` | KNN, Decision Tree, SVM |
| 03 | `03_regression_housing.ipynb` | Linear, Ridge, Lasso, Random Forest |
| 04 | `04_clustering_kmeans.ipynb` | K-Means, DBSCAN, Elbow method |
| 05 | `05_neural_network_mnist.ipynb` | PyTorch 3-layer NN on MNIST |

---

## 3. Reusable Utilities

Import from `src/` in any notebook:

```python
import sys
sys.path.insert(0, 'src')

from preprocessor import DataPreprocessor
from evaluator    import compare_models, evaluate_classifier
from visualizer   import plot_confusion_matrix, plot_training_curves
```

---

## 4. Run Tests

```bash
pytest tests/ -v
```

Expected output: all 8 tests passing ✅

---

## 5. GPU Training (Notebook 05)

The MNIST notebook auto-detects CUDA. If you have a compatible GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If CPU only, training takes ~3–5 min for 15 epochs.

---

## 6. Project Structure

```
ml-mini-projects/
├── 01_data_preprocessing.ipynb
├── 02_classification_iris.ipynb
├── 03_regression_housing.ipynb
├── 04_clustering_kmeans.ipynb
├── 05_neural_network_mnist.ipynb
├── src/
│   ├── preprocessor.py      # DataPreprocessor class
│   ├── evaluator.py         # compare_models, evaluate_*
│   └── visualizer.py        # All plotting utilities
├── tests/
│   ├── test_preprocessor.py
│   └── test_evaluator.py
├── data/raw/                # Datasets (gitignored large files)
├── models/                  # Saved model weights
├── outputs/figures/         # Plot exports
├── docs/SETUP.md
├── .github/workflows/ml-ci.yml
├── requirements.txt
├── setup.py
└── .gitignore
```
