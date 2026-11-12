# 📊 Week 9 & Week 10: Machine Learning Model Evaluation, Feature Engineering & Clustering Pipelines

An enterprise-grade, dual-track machine learning laboratory exploring advanced **Supervised Classification & Model Evaluation** alongside **Unsupervised Customer Segmentation & Clustering Pipelines**. This repository demonstrates production-ready preprocessing pipelines, leakage-free validation, feature engineering, and rigorous metric benchmarking using `scikit-learn`.

---

## 📌 Project Overview

This module covers two comprehensive machine learning workflows for **Week 9 & Week 10**:

1. **Part 1: Housing Price Classification & Advanced Model Evaluation**
   - **Task**: Predict whether a residential property falls into the **High-Value** category ($\ge \text{Median Price} = \$4,340,000$) or **Standard-Value** category.
   - **Feature Engineering**: Constructed domain-specific ratios (`area_per_bedroom`, `area_per_room`), composite luxury indices (`luxury_index`), and total amenity scores (`amenities_score`).
   - **Feature Scaling Benchmark**: Comparative analysis between **StandardScaler** ($Z$-score), **MinMaxScaler** ($[0, 1]$ bounding), and **RobustScaler** (interquartile range median-centering).
   - **Validation Strategy**: 5-Fold Stratified Cross-Validation combined with an independent hold-out test set (80/20 stratified split).
   - **Evaluation Diagnostics**: Multi-model assessment using Confusion Matrices, Precision-Recall Curves (PR-AUC), Receiver Operating Characteristic Curves (ROC-AUC), and Random Forest Gini Feature Importances.

2. **Part 2: Customer Segmentation & Discount Targeting (`w9&w10/`)**
   - **Task**: Segment 1,000 e-commerce customer browsing and purchasing profiles into distinct marketing personas for targeted promotional discounting.
   - **Pipeline Design**: Leakage-free `ColumnTransformer` integrating numerical standardization, categorical one-hot encoding, and binary passthrough.
   - **Clustering Algorithms**: Benchmark between **K-Means** (Elbow & Silhouette analysis), **Hierarchical Agglomerative** (Ward linkage dendrograms), and **DBSCAN** ($k$-distance graph calibration and 2D PCA spatial density clustering).
   - **Unsupervised Evaluation**: Quantitative validation via **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.
   - **Business Personas**: Data-driven customer profiling heatmap mapping conversion rates, session length, and cart values to optimal discount eligibility.

---

## 📁 Directory Architecture

```
Week 9 & Week 10/
├── Housing.csv                                      # Housing price dataset (545 rows, 13 features)
├── model_evaluation_and_feature_engineering.py     # Python script executing housing ML pipeline
├── Housing_ML_Evaluation_and_Feature_Engineering.ipynb # Interactive Jupyter Notebook for housing ML
├── generate_notebook.py                            # Notebook generator utility
├── README.md                                       # Comprehensive module documentation
├── plots/                                          # Visual diagnostics for Part 1
│   ├── scaling_comparison.png                      # Distribution plots across 3 scalers
│   ├── cv_scores_comparison.png                    # 5-fold CV metric comparison across 5 models
│   ├── confusion_matrices.png                      # Confusion matrices on test set
│   ├── roc_auc_curves.png                          # ROC curves and AUC scores
│   ├── precision_recall_curves.png                 # Precision-Recall curves & Average Precision
│   └── feature_importance.png                      # Top 10 Random Forest feature importances
│
└── w9&w10/                                         # Part 2: Unsupervised Customer Segmentation
    ├── customer_segmentation_with_clusters.csv     # Final dataset enriched with cluster labels
    ├── clustering_workflow.py                      # Python script for clustering pipeline
    ├── clustering_workflow.ipynb                   # Interactive Jupyter Notebook for clustering
    ├── archive (3)/
    │   └── customer_segmentation_discount.csv      # Source customer segmentation dataset (1000 rows)
    └── output_plots/                               # Visual diagnostics for Part 2
        ├── kmeans_elbow_silhouette.png             # WCSS Elbow & Silhouette score vs k
        ├── hierarchical_dendrogram.png             # SciPy Ward linkage dendrogram
        ├── dbscan_kdistance.png                    # 5-NN distance graph for epsilon selection
        ├── clusters_comparison_pca.png             # 2D PCA projection of K-Means, Agglomerative & DBSCAN
        └── cluster_profiles_heatmap.png            # Normalized customer segment feature heatmap
```

---

## 🔬 Part 1: Housing Price Classification & Model Evaluation

### 1. Data Preprocessing & Feature Engineering
- **Simulated Robust Imputation**: Demonstrated median imputation for numerical data (`area`) and mode imputation for categorical data (`furnishingstatus`).
- **Feature Synthesis**:
  - `total_rooms = bedrooms + bathrooms`
  - `area_per_bedroom = area / bedrooms`
  - `area_per_room = area / total_rooms`
  - `amenities_score = mainroad + guestroom + basement + hotwater + airconditioning + prefarea + parking_flag`
  - `luxury_index = area * bathrooms * (airconditioning + 1)`
- **Scaling Comparison**: `StandardScaler`, `MinMaxScaler`, and `RobustScaler` evaluated across skewed features (`area`, `area_per_bedroom`, `luxury_index`).

### 2. Experimental Results & Benchmark (Housing Dataset)

Evaluated with an 80/20 Stratified Train-Test Split (436 train, 109 test) and **5-Fold Stratified Cross-Validation**:

| Model | 5-Fold CV Acc | 5-Fold CV F1 | 5-Fold CV ROC-AUC | Test Acc | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8372** | **0.8340** | 0.9104 | **0.8624** | 0.8571 | **0.8727** | **0.8649** | **0.9414** |
| **Gradient Boosting** | 0.8143 | 0.8120 | 0.9019 | 0.8440 | **0.8800** | 0.8000 | 0.8381 | 0.8966 |
| **Random Forest** | 0.8073 | 0.8023 | **0.9107** | 0.8349 | 0.8776 | 0.7818 | 0.8269 | 0.9205 |
| **Support Vector Classifier (RBF)** | 0.8005 | 0.7961 | 0.8943 | 0.8073 | 0.8696 | 0.7273 | 0.7921 | 0.9320 |
| **K-Nearest Neighbors ($k=7$)** | 0.7798 | 0.7793 | 0.8771 | 0.7890 | 0.8333 | 0.7273 | 0.7767 | 0.8764 |

### 3. Key Observations & Takeaways (Classification)
1. **Linear Superiority**: **Logistic Regression** achieved the highest overall Test Accuracy (**86.24%**), Test F1 (**0.8649**), and Test ROC-AUC (**0.9414**). The engineered composite interaction features provided linear separability that simple tree depths did not capture as smoothly.
2. **Feature Importance Hierarchy**: Random Forest Gini feature importances confirmed that `luxury_index`, `area`, `area_per_room`, and `amenities_score` dominated predictive value over raw bedroom counts.
3. **Threshold Calibration**: Precision-Recall curves indicate an Average Precision (AP) exceeding **0.93** for Logistic Regression and Random Forest, providing high reliability in ranking premium real estate.

---

## 🛍️ Part 2: Customer Segmentation & Unsupervised Clustering

### 1. Preprocessing & Feature Breakdown
- **Dataset**: 1,000 customers $\times$ 24 behavioral and transactional variables.
- **Continuous Features (11)**: `age`, `annual_income`, `current_session_duration_sec`, `pages_viewed`, `cart_total_value`, `lifetime_purchases_count`, `customer_conversion_rate`, etc.
- **Categorical Features (3)**: `traffic_source`, `product_category`, `previous_review_sentiment`.
- **Binary Features (8)**: `device_mobile`, `basket_icon_click`, `scrolled_to_reviews`, `saw_checkout`, `is_returning_user`, etc.

### 2. Clustering Benchmark & Validation Metrics

| Algorithm | Clusters ($k$) | Outliers / Noise | Silhouette Score (Higher $\uparrow$) | Davies-Bouldin Index (Lower $\downarrow$) | Calinski-Harabasz Index (Higher $\uparrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=3$, `k-means++`)** | 3 | 0 (0.0%) | **0.1608** | **1.8508** | **182.95** |
| **Hierarchical Agglomerative (Ward)** | 3 | 0 (0.0%) | 0.1477 | 2.0400 | 153.90 |
| **DBSCAN ($\varepsilon=2.8, \text{min}=4$)** | 2 | 136 (13.6%) | 0.0926 | 1.4437 | 6.87 |
| **DBSCAN (2D PCA, $\varepsilon=0.55$)** | 2 | 22 (2.2%) | **0.4873** | **0.8695** | **938.29** |

### 3. Customer Segments & Business Strategy
- **Cluster 0 — High-Intent Window Shoppers (38.6%)**: High session duration and page views, but moderate conversion rate. Responds favorably to **time-sensitive discount coupons** to trigger conversion.
- **Cluster 1 — Low-Engagement Casual Visitors (41.1%)**: Short visit duration, few items in cart. Best addressed via introductory promotions and onboarding banners.
- **Cluster 2 — VIP High-Value Frequent Buyers (20.3%)**: Top lifetime purchase count and highest cart total value. Do not require aggressive discounts; benefit most from **loyalty rewards** and premium perks.

---

## 🛠️ Installation & Execution

### Prerequisites
- Python 3.9+
- Standard ML stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Running Part 1 (Housing ML Evaluation)
```bash
# Execute standalone script (generates terminal metrics and figures in plots/)
python model_evaluation_and_feature_engineering.py
```
Or open and execute `Housing_ML_Evaluation_and_Feature_Engineering.ipynb` in VS Code or JupyterLab.

### Running Part 2 (Clustering Pipeline)
```bash
# Navigate to clustering directory
cd "w9&w10"

# Execute clustering workflow (outputs metrics and plots in output_plots/)
python clustering_workflow.py
```
Or open and interact with `w9&w10/clustering_workflow.ipynb`.

---

## 👥 Author & Academic Context
- **Course**: Problem Based Learning (PBL) Hack O Week — 5th Semester
- **Focus**: Feature Engineering, Metric Rigor, Scalers Comparison, Cross-Validation, and Enterprise Clustering Pipelines.
