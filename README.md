# 🤖 ML Toolbox — Modular AI Agent System

A complete, production-ready machine learning pipeline built with **10 specialized AI agents**, supporting datasets of all sizes (n < 30 to n > 10,000).

---

## 📁 Folder Structure

```
ml_toolbox/
├── app.py                          # Streamlit web interface
├── pipeline.py                     # Pipeline orchestrator
├── requirements.txt
├── demo.py                         # Programmatic demo script
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── data_understanding_agent.py   # EDA, missing values, outliers, correlation
│   ├── preprocessing_agent.py        # Imputation, encoding, scaling
│   ├── feature_engineering_agent.py  # MI, RFE, polynomial features
│   ├── small_dataset_agent.py        # LOO-CV, bootstrap, regularization
│   ├── model_training_agent.py       # 7 ML algorithms with CV
│   ├── hyperparameter_agent.py       # Grid/Random/Bayesian search
│   ├── evaluation_agent.py           # Accuracy, F1, ROC-AUC, etc.
│   ├── visualization_agent.py        # Confusion matrix, ROC, heatmaps
│   ├── explainability_agent.py       # SHAP + LIME
│   └── research_paper_agent.py       # Auto research paper generation
│
└── data/
    ├── __init__.py
    └── example_datasets.py           # 5 built-in example datasets
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Web Interface

```bash
streamlit run app.py
```

### 3. Run Demo (Programmatic)

```bash
python demo.py
```

---

## 🤖 The 10 AI Agents

| # | Agent | Purpose |
|---|-------|---------|
| 1 | **Data Understanding** | Dataset summary, missing values, outlier detection, correlation analysis |
| 2 | **Preprocessing** | Imputation (KNN/Simple), encoding (OHE/Label), scaling (Standard/Robust/MinMax) |
| 3 | **Feature Engineering** | Mutual information, RFE, polynomial features |
| 4 | **Small Dataset Optimization** | LOO-CV, bootstrapping, regularization analysis, feature-to-sample ratio |
| 5 | **Model Training** | 7 algorithms: LR, SVM, DT, RF, GB, NB, KNN |
| 6 | **Hyperparameter Optimization** | Grid Search, Random Search, Bayesian Optimization |
| 7 | **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, MCC, Cohen's Kappa |
| 8 | **Visualization** | Confusion matrix, ROC curves, feature importance, correlation heatmap |
| 9 | **Explainability** | SHAP (TreeExplainer + KernelExplainer) + LIME |
| 10 | **Research Paper** | Auto-generates methodology, experimental setup, results, discussion, LaTeX tables |

---

## 📊 Dataset Compatibility

| Dataset Size | CV Strategy | Recommended Models |
|---|---|---|
| n < 30 | Leave-One-Out CV | LR, NB, SVM |
| 30 ≤ n < 100 | Stratified 5-Fold | All models |
| n ≥ 100 | Stratified 10-Fold | All models |

---

## 💻 Programmatic Usage

```python
from pipeline import MLPipeline
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Initialize pipeline
pipeline = MLPipeline(output_dir="./outputs")

# Run complete pipeline
results = pipeline.run(
    df=df,
    target_col="your_target",
    dataset_name="My Dataset",
    task_type="classification",      # or "regression"
    top_k_features=10,
    hp_method="random",              # "grid", "random", or "bayesian"
    hp_n_iter=20,
    run_explainability=True,
    run_hp_optimization=True,
)

# Access results
print(f"Best Model: {results['evaluation']['best_model']}")

# Get model comparison
comparison_df = pipeline.get_model_comparison_df()
print(comparison_df)

# Get research paper text
paper = pipeline.get_research_paper_text()

# Save all results to JSON
pipeline.save_results()
```

---

## 📦 Built-in Example Datasets

| Dataset | Samples | Task | Features |
|---|---|---|---|
| 🌸 Iris | 150 | Multiclass | 4 numeric |
| 🎗️ Breast Cancer | 569 | Binary | 30 numeric |
| 🍷 Wine | 178 | Multiclass | 13 numeric |
| 🔬 Small Synthetic | 60 | Binary | Mixed (numeric + categorical) |
| 🌙 Moons | 200 | Binary | 2 numeric (non-linear) |

---

## 🛠️ Tech Stack

- **Core ML:** scikit-learn, scipy
- **Data:** pandas, numpy
- **Explainability:** SHAP, LIME
- **Visualization:** matplotlib, seaborn
- **HP Optimization:** scikit-optimize (Bayesian)
- **Web Interface:** Streamlit
- **Imbalanced Data:** imbalanced-learn

---

## 📄 Research Paper Output

The Research Paper Agent automatically generates:
- **Abstract** — Dataset and findings summary
- **Methodology** — Data preprocessing, feature engineering, model development
- **Experimental Setup** — Hardware, software, dataset characteristics
- **Results** — Comparative performance table (also as LaTeX)
- **Discussion** — Model selection rationale, SHAP insights, limitations
- **Conclusion** — Summary and future work
- **BibTeX References** — Standard ML citations

All downloadable as a Markdown file from the web interface.

---

## 🔧 Configuration

The pipeline auto-selects optimal settings based on dataset size, but all settings are configurable:

```python
# Custom preprocessing config
preprocessing_config = {
    "imputation": {
        "numeric_method": "knn",         # "simple" or "knn"
        "categorical_method": "most_frequent",
    },
    "outlier_handling": {
        "method": "iqr_clip",            # "iqr_clip", "zscore_remove", "none"
        "threshold": 1.5,
    },
    "encoding": {
        "method": "onehot",              # "onehot" or "label"
        "drop_first": True,
    },
    "scaling": {
        "method": "robust",              # "standard", "minmax", "robust"
    },
}

results = pipeline.run(df, target_col, preprocessing_config=preprocessing_config)
```
