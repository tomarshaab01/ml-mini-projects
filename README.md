# 🧠 ML Mini Projects

> A structured collection of **5 end-to-end Machine Learning notebooks** in Python — from raw data preprocessing to neural network training — with reusable utilities, full test coverage, and automated CI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikitlearn) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch) ![CI](https://github.com/tomarshaab01/ml-mini-projects/actions/workflows/ml-ci.yml/badge.svg) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📚 Notebooks

| # | Notebook | Algorithms | Dataset |
|---|----------|------------|---------|
| 01 | [Data Preprocessing](01_data_preprocessing.ipynb) | Imputation · LabelEncoder · StandardScaler | Synthetic HR data |
| 02 | [Classification](02_classification_iris.ipynb) | KNN · Decision Tree · SVM + CV | Iris |
| 03 | [Regression](03_regression_housing.ipynb) | Linear · Ridge · Lasso · Random Forest | California Housing |
| 04 | [Clustering](04_clustering_kmeans.ipynb) | K-Means · DBSCAN · Elbow · Silhouette | Synthetic customers |
| 05 | [Neural Network](05_neural_network_mnist.ipynb) | PyTorch FC-Net · BatchNorm · Dropout · Adam | MNIST |

---

## 🗂️ Project Structure

```
ml-mini-projects/
├── 01–05 *.ipynb            ← Notebooks (run in order)
├── src/
│   ├── preprocessor.py      ← DataPreprocessor class
│   ├── evaluator.py         ← compare_models, evaluate_classifier/regressor
│   └── visualizer.py        ← confusion matrix, feature importance, training curves, clusters
├── tests/
│   ├── test_preprocessor.py ← 4 unit tests
│   └── test_evaluator.py    ← 4 unit tests
├── docs/SETUP.md            ← Full setup & usage guide
├── .github/workflows/       ← CI: lint → test → validate notebooks → stats
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/tomarshaab01/ml-mini-projects.git
cd ml-mini-projects
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## 🔁 Reusable Utilities (`src/`)

```python
import sys; sys.path.insert(0, 'src')

from preprocessor import DataPreprocessor
from evaluator    import compare_models
from visualizer   import plot_confusion_matrix, plot_training_curves

# One-line preprocessing
X_train, X_test, y_train, y_test = DataPreprocessor().fit_transform(df, target_col='label')

# One-line model comparison
results = compare_models({'RF': rf_model, 'SVM': svm_model}, X_train, X_test, y_train, y_test)
```

---

## 🤖 GitHub Actions CI

Every push triggers 4 automated jobs:

| Job | What it checks |
|-----|----------------|
| **Lint** | `flake8` on all `src/` and `tests/` Python files |
| **Test** | `pytest` — 8 unit tests across preprocessor & evaluator |
| **Validate Notebooks** | All `.ipynb` files are valid JSON with correct structure |
| **Code Stats** | Line counts per file (informational) |

---

## 📊 Results Summary

| Task | Best Model | Metric |
|------|-----------|--------|
| Classification (Iris) | SVM (RBF) | ~97% accuracy |
| Regression (Housing) | Random Forest | R² ~0.80 |
| Clustering (Customers) | K-Means (k=3) | Silhouette ~0.55 |
| Neural Network (MNIST) | FC-Net (PyTorch) | ~98.5% accuracy |

---

## 👨‍💻 Author

**Bharat Tomar** · B.Tech AI & ML @ AKGEC-AKTU  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Bharat_Tomar-blue?logo=linkedin)](https://www.linkedin.com/in/bharat-tomar-026a87366) [![GitHub](https://img.shields.io/badge/GitHub-tomarshaab01-black?logo=github)](https://github.com/tomarshaab01)
