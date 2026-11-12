# 📊 Week 7 & Week 8: Machine Learning Models Analysis & Salary Prediction

An end-to-end Machine Learning benchmark and comparative study exploring **Regression** and **Classification** algorithms applied to an employee salary dataset. This project evaluates model fit, generalization performance, regularization effects, and classification metrics using `scikit-learn`.

---

## 📌 Project Overview

This repository contains the practical experiments and analysis conducted for **Week 7 & Week 8**:
1. **Regression Analysis**: Continuous prediction of employee monthly salaries using linear and regularized estimators.
2. **Classification Analysis**: Binary prediction of salary category (`High Salary` vs. `Low/Average Salary`, split by median threshold $\text{Salary} > ₹73,890.50$) to evaluate decision boundary and neighbor-based performance.

---

## 📂 Project Architecture

```
Week 7 & Week 8/
├── employee_salary_dataset.csv       # Dataset with 50 employee records & 9 attributes
├── ml_models_analysis.py             # Full Python script executing pipelines, metrics & plots
├── Model_Experiments.ipynb           # Interactive Jupyter Notebook for experiments
├── README.md                         # Project documentation and results summary
├── visualizations/                   # Generated evaluation plots
│   ├── regression_predictions.png
│   └── classification_confusion_matrices.png
└── extracted/                        # Extracted source data files
```

---

## 📊 Dataset Details

- **Records**: 50 employees
- **Target (Regression)**: `Monthly_Salary` (Continuous, Min: ₹28,420, Max: ₹149,123, Mean: ₹82,288.80)
- **Target (Classification)**: `High_Salary` (Binary: `0` = Low/Average, `1` = High, Balanced 25:25)
- **Numerical Features**: `Experience_Years`, `Age` (StandardScaler applied)
- **Categorical Features**: `Department`, `Education_Level`, `Gender`, `City` (OneHotEncoded)

---

## 🔬 Experimental Results

### 1. Regression Models (Predicting `Monthly_Salary`)

Evaluated with an 80/20 train-test split (`random_state=42`):

| Model | Train $R^2$ | Test $R^2$ | MAE (₹) | RMSE (₹) | Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | 0.2123 | **0.2178** | **24,918.86** | **30,368.18** | Baseline linear model; best generalizer on unseen test data. |
| **Polynomial Regression (Degree 2)** | **0.4736** | -0.0732 | 28,378.38 | 35,571.56 | Shows significant overfitting due to feature explosion on limited sample size. |
| **Ridge Regression ($\alpha=1.0$)** | 0.1991 | 0.0938 | 28,207.13 | 32,686.78 | L2 regularization shrinks weights to stabilize variance. |
| **Lasso Regression ($\alpha=500.0$)** | 0.1872 | 0.0442 | 28,889.02 | 33,570.21 | L1 regularization performs feature selection and drives smaller coefficients to zero. |

---

### 2. Classification Models (Predicting `High_Salary`)

Evaluated using a Stratified 80/20 split (`random_state=42`):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 60.00% | 57.14% | 80.00% | 0.6667 | 0.5200 |
| **K-Nearest Neighbors ($k=5$)** | 60.00% | 57.14% | 80.00% | 0.6667 | **0.6600** |
| **K-Nearest Neighbors ($k=3$)** | 60.00% | 57.14% | 80.00% | 0.6667 | 0.5800 |

#### Confusion Matrix (Classification Test Set, $n=10$)
```
[[True Negative (TN) = 2,  False Positive (FP) = 3],
 [False Negative (FN) = 1,  True Positive (TP) = 4]]
```

---

## 📈 Visualizations

Visual plots are automatically exported to the `visualizations/` folder:
- **`regression_predictions.png`**: Actual vs. Predicted salary values compared across Linear, Polynomial, Ridge, and Lasso estimators.
- **`classification_confusion_matrices.png`**: Heatmaps representing True vs. Predicted classifications for Logistic Regression and KNN.

---

## 🚀 How to Run

### 1. Prerequisites
Install required Python packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. Run the Analysis Script
```bash
python ml_models_analysis.py
```

### 3. Open the Interactive Notebook
```bash
jupyter notebook Model_Experiments.ipynb
```
