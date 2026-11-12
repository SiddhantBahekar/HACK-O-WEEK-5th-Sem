import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Machine Learning Pipeline: Feature Engineering, Scaling, Missing Data & Model Evaluation\n",
    "**Dataset:** Housing Price Prediction Dataset (`Housing.csv`)\n",
    "\n",
    "---\n",
    "\n",
    "## 🎯 Objectives\n",
    "1. **Data Preprocessing & Missing Data Handling**: Techniques for dealing with missing values in numerical and categorical features.\n",
    "2. **Feature Engineering & Feature Scaling**: Domain-specific feature construction and comparing `StandardScaler`, `MinMaxScaler`, and `RobustScaler`.\n",
    "3. **Model Evaluation & Validation Strategy**:\n",
    "   - **Train/Test Split** with stratification to prevent data leakage.\n",
    "   - **Stratified K-Fold Cross-Validation** (5-Fold CV) across multiple algorithms.\n",
    "   - **Confusion Matrix** analysis (True Positives, False Positives, True Negatives, False Negatives).\n",
    "   - **Precision, Recall, F1-Score, and Support** metrics.\n",
    "   - **Receiver Operating Characteristic (ROC-AUC)** and **Precision-Recall (PR-AUC)** curves."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Import Libraries and Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "from sklearn.metrics import (\n",
    "    confusion_matrix, ConfusionMatrixDisplay, classification_report,\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score\n",
    ")\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Load Dataset and Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('Housing.csv')\n",
    "print(f\"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Handling Missing Data\n",
    "\n",
    "Missing values are common in real-world tabular data. Strategies include:\n",
    "- **Numerical Data:** Mean imputation (normal distribution) or Median imputation (skewed distribution / outliers).\n",
    "- **Categorical Data:** Mode (most frequent) imputation or a dedicated `'Missing'` category tag."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checking existing missing values\n",
    "print(\"Missing values count per column:\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "# Imputation pipeline example\n",
    "num_imputer = SimpleImputer(strategy='median')\n",
    "cat_imputer = SimpleImputer(strategy='most_frequent')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Feature Engineering\n",
    "\n",
    "We create domain-specific features from raw columns:\n",
    "1. **`total_rooms`**: Sum of bedrooms and bathrooms.\n",
    "2. **`area_per_bedroom`**: Space density per bedroom.\n",
    "3. **`area_per_room`**: Space density across all rooms.\n",
    "4. **`amenities_score`**: Total number of premium amenities present.\n",
    "5. **`luxury_index`**: Interaction term (`area * bathrooms * airconditioning`).\n",
    "6. **Target Variable**: `high_value_target` binary indicator (Price $\\ge$ Median Price)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_fe = df.copy()\n",
    "\n",
    "# 1. Total rooms\n",
    "df_fe['total_rooms'] = df_fe['bedrooms'] + df_fe['bathrooms']\n",
    "\n",
    "# 2. Area per bedroom\n",
    "df_fe['area_per_bedroom'] = df_fe['area'] / df_fe['bedrooms']\n",
    "\n",
    "# 3. Area per room\n",
    "df_fe['area_per_room'] = df_fe['area'] / df_fe['total_rooms']\n",
    "\n",
    "# 4. Binary amenity encoding\n",
    "binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']\n",
    "for col in binary_cols:\n",
    "    df_fe[col + '_code'] = (df_fe[col].str.lower() == 'yes').astype(int)\n",
    "\n",
    "# 5. Amenities Score\n",
    "df_fe['amenities_score'] = (\n",
    "    df_fe['mainroad_code'] + df_fe['guestroom_code'] + df_fe['basement_code'] +\n",
    "    df_fe['hotwaterheating_code'] + df_fe['airconditioning_code'] + df_fe['prefarea_code'] +\n",
    "    (df_fe['parking'] > 0).astype(int)\n",
    ")\n",
    "\n",
    "# 6. Luxury Index Interaction\n",
    "df_fe['luxury_index'] = df_fe['area'] * df_fe['bathrooms'] * (df_fe['airconditioning_code'] + 1)\n",
    "\n",
    "# 7. Classification Target\n",
    "price_median = df_fe['price'].median()\n",
    "df_fe['high_value_target'] = (df_fe['price'] >= price_median).astype(int)\n",
    "\n",
    "print(f\"Target distribution:\\n{df_fe['high_value_target'].value_counts(normalize=True)}\")\n",
    "df_fe[['area', 'total_rooms', 'amenities_score', 'luxury_index', 'high_value_target']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Feature Scaling Comparison (StandardScaler vs MinMaxScaler vs RobustScaler)\n",
    "\n",
    "- **StandardScaler**: $z = \\frac{x - \\mu}{\\sigma}$ (Zero mean, unit variance. Best for algorithms assuming Gaussian distributions like Logistic Regression, SVM).\n",
    "- **MinMaxScaler**: $x' = \\frac{x - x_{min}}{x_{max} - x_{min}} \\in [0, 1]$ (Bounded scale. Sensitive to outliers).\n",
    "- **RobustScaler**: $x' = \\frac{x - Q_2}{IQR}$ (Uses median and Interquartile Range; resilient to extreme outliers)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_features_to_scale = ['area', 'area_per_bedroom', 'luxury_index']\n",
    "\n",
    "scaled_std = StandardScaler().fit_transform(df_fe[num_features_to_scale])\n",
    "scaled_minmax = MinMaxScaler().fit_transform(df_fe[num_features_to_scale])\n",
    "scaled_robust = RobustScaler().fit_transform(df_fe[num_features_to_scale])\n",
    "\n",
    "fig, axes = plt.subplots(3, 4, figsize=(16, 10))\n",
    "scalers_dict = {\n",
    "    'Original': df_fe[num_features_to_scale].values,\n",
    "    'StandardScaler': scaled_std,\n",
    "    'MinMaxScaler': scaled_minmax,\n",
    "    'RobustScaler': scaled_robust\n",
    "}\n",
    "\n",
    "for row_idx, feat in enumerate(num_features_to_scale):\n",
    "    for col_idx, (s_name, s_data) in enumerate(scalers_dict.items()):\n",
    "        ax = axes[row_idx, col_idx]\n",
    "        sns.histplot(s_data[:, row_idx], kde=True, ax=ax, color='#2b5c8f' if col_idx==0 else '#207561')\n",
    "        ax.set_title(f\"{feat} ({s_name})\", fontsize=10, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 6: Train/Test Split & Preprocessing Pipeline\n",
    "\n",
    "To prevent data leakage, all imputation and scaling transformations are fit **strictly** on the training split and applied to the test split via `sklearn.pipeline.Pipeline`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',\n",
    "                  'total_rooms', 'area_per_bedroom', 'area_per_room', 'amenities_score', 'luxury_index']\n",
    "categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']\n",
    "\n",
    "X = df_fe[numerical_cols + categorical_cols]\n",
    "y = df_fe['high_value_target']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "\n",
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
    "    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numerical_cols),\n",
    "        ('cat', categorical_transformer, categorical_cols)\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 7: Cross-Validation (5-Fold Stratified CV)\n",
    "\n",
    "We evaluate multiple machine learning classifiers with 5-fold cross-validation across key metrics:\n",
    "- **Accuracy**\n",
    "- **Precision**\n",
    "- **Recall**\n",
    "- **F1-Score**\n",
    "- **ROC-AUC**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),\n",
    "    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),\n",
    "    'Support Vector Classifier': SVC(probability=True, random_state=42),\n",
    "    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)\n",
    "}\n",
    "\n",
    "cv_scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}\n",
    "cv_summary = []\n",
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "for name, model in models.items():\n",
    "    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])\n",
    "    scores = cross_validate(pipe, X_train, y_train, cv=skf, scoring=cv_scoring)\n",
    "    cv_summary.append({\n",
    "        'Model': name,\n",
    "        'Accuracy (Mean)': scores['test_accuracy'].mean(),\n",
    "        'Precision (Mean)': scores['test_precision'].mean(),\n",
    "        'Recall (Mean)': scores['test_recall'].mean(),\n",
    "        'F1-Score (Mean)': scores['test_f1'].mean(),\n",
    "        'ROC-AUC (Mean)': scores['test_roc_auc'].mean()\n",
    "    })\n",
    "\n",
    "cv_df = pd.DataFrame(cv_summary)\n",
    "cv_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 8: Hold-Out Test Set Evaluation (Confusion Matrix, Precision/Recall/F1, ROC-AUC)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 9))\n",
    "axes_cm = axes_cm.flatten()\n",
    "\n",
    "fig_roc, ax_roc = plt.subplots(figsize=(8, 6))\n",
    "fig_pr, ax_pr = plt.subplots(figsize=(8, 6))\n",
    "\n",
    "test_results = []\n",
    "\n",
    "for idx, (name, model) in enumerate(models.items()):\n",
    "    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])\n",
    "    pipe.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred = pipe.predict(X_test)\n",
    "    y_proba = pipe.predict_proba(X_test)[:, 1]\n",
    "    \n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    prec = precision_score(y_test, y_pred)\n",
    "    rec = recall_score(y_test, y_pred)\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    roc_auc = roc_auc_score(y_test, y_proba)\n",
    "    ap = average_precision_score(y_test, y_proba)\n",
    "    \n",
    "    test_results.append({\n",
    "        'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc_auc, 'PR-AUC': ap\n",
    "    })\n",
    "    \n",
    "    # Confusion Matrix\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Standard', 'High-Value'])\n",
    "    disp.plot(ax=axes_cm[idx], cmap='Blues', colorbar=False)\n",
    "    axes_cm[idx].set_title(f\"{name}\\n(F1: {f1:.3f})\", fontweight='bold')\n",
    "    axes_cm[idx].grid(False)\n",
    "    \n",
    "    # ROC Curve\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_proba)\n",
    "    ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')\n",
    "    \n",
    "    # Precision-Recall Curve\n",
    "    p_curve, r_curve, _ = precision_recall_curve(y_test, y_proba)\n",
    "    ax_pr.plot(r_curve, p_curve, lw=2, label=f'{name} (AP = {ap:.3f})')\n",
    "\n",
    "if len(models) < len(axes_cm):\n",
    "    fig_cm.delaxes(axes_cm[-1])\n",
    "fig_cm.tight_layout()\n",
    "\n",
    "ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (0.50)')\n",
    "ax_roc.set_title('ROC Curves', fontweight='bold')\n",
    "ax_roc.set_xlabel('False Positive Rate')\n",
    "ax_roc.set_ylabel('True Positive Rate')\n",
    "ax_roc.legend(loc='lower right')\n",
    "\n",
    "ax_pr.plot([0, 1], [0.5, 0.5], 'k--', lw=1.5, label='Baseline (0.50)')\n",
    "ax_pr.set_title('Precision-Recall Curves', fontweight='bold')\n",
    "ax_pr.set_xlabel('Recall')\n",
    "ax_pr.set_ylabel('Precision')\n",
    "ax_pr.legend(loc='lower left')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.DataFrame(test_results)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("Housing_ML_Evaluation_and_Feature_Engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Created Housing_ML_Evaluation_and_Feature_Engineering.ipynb successfully.")
