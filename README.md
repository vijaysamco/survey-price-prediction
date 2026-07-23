# Consumer Survey Price Range Prediction

This repository contains the complete Machine Learning workflow for analyzing consumer survey responses, cleaning data, engineering domain-specific features, and predicting price range preferences using multi-class classification models.

Experiment tracking, model registry, and metric versioning are fully integrated using **MLflow** and hosted on **DagsHub**.

---

# 📁 Repository Structure

```text
survey-price-prediction/
├── data/
│   ├── survey_results.csv
│   ├── cleaned_survey_results.csv
│   └── engineered_survey_results.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_predictive_modeling.ipynb
├── src/
│   └── train_and_track.py
└── app.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Data Pipeline

## 1. Data Cleaning

### Duplicate Removal
- Removed **10** exact duplicate records.

### Outlier Detection
- Removed **9** extreme human age outliers (values greater than **100**, e.g. 192, 285, 604).

### Missing Value Imputation

| Column | Strategy |
|---------|----------|
| income_levels | Filled with **"Not Reported"** |
| consume_frequency(weekly) | Filled using mode (**3–4 times**) |
| purchase_channel | Filled using mode (**Online**) |

### Spelling Standardization

**Zone**

| Incorrect | Correct |
|-----------|---------|
| urbna | Urban |
| Metor | Metro |

**Current Brand**

| Incorrect | Correct |
|-----------|---------|
| newcomer | Newcomer |
| Establishd | Established |

---

# ⚙️ Feature Engineering

## Age Group

Converted numerical age into demographic categories:

- 18–25
- 26–35
- 36–45
- 46–55
- 56–70
- 70+

The original **Age** column was removed.

---

## CF-AB Score

Formula

\[
CF\_AB = \frac{Frequency\ Score}{Awareness\ Score + Frequency\ Score}
\]

Measures interaction between consumption frequency and brand awareness.

---

## Zone Affluence Score (ZAS)

Formula

\[
ZAS = Zone\ Score \times Income\ Score
\]

Measures geographical purchasing power.

---

## Brand Switching Indicator (BSI)

Binary feature:

- **1** → Consumer uses a non-established brand due to Price or Quality.
- **0** → Otherwise.

---

## Logical Outlier Removal

Removed **35** inconsistent records where:

- Occupation = Student
- Age Group = 56–70

---

# 🏆 Model Evaluation

Train/Test Split:

- **75% Training**
- **25% Testing**
- **random_state = 42**

Target Classes:

- 50–100
- 100–150
- 150–200
- 200–250

| Rank | Model | Accuracy | Weighted F1 | Status |
|------|--------|----------|-------------|--------|
| 1 | XGBoost | **92.28%** | **0.92** | Best Model |
| 2 | Random Forest | 89.09% | 0.89 | High Performing |
| 3 | SVM | 82.45% | 0.82 | Baseline |
| 4 | Logistic Regression | 80.09% | 0.80 | Baseline |
| 5 | Gaussian Naive Bayes | 58.31% | 0.55 | Baseline |

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone https://dagshub.com/YOUR_USERNAME/survey-price-prediction.git

cd survey-price-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python src/train_and_track.py
```

This script will:

- Load engineered dataset
- Perform Label Encoding
- Perform One-Hot Encoding
- Train all ML models
- Evaluate metrics
- Track experiments using MLflow
- Upload models to DagsHub

---

# 📊 Experiment Tracking

Each experiment automatically logs:

- Parameters
- Metrics
- Model artifacts
- Confusion matrices
- Feature importance
- Model versions

All runs can be viewed inside your DagsHub repository under the **Experiments** tab.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- MLflow
- DagsHub
- Jupyter Notebook

---

# 👨‍💻 Author

**Vijay**

Machine Learning | Data Science | MLOps