import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score,
    cross_validate
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# -------------------------------------------------------------
# 0. Configuration & Setup
# -------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATASET_PATH = "Housing.csv"
print(f"Loading dataset from: {DATASET_PATH} ...")
df = pd.read_csv(DATASET_PATH)

print("\n" + "="*80)
print("1. DATASET OVERVIEW & EXPLORATION")
print("="*80)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types & Non-Null Counts:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
missing_counts = df.isnull().sum()
print("\nMissing values per column:")
print(missing_counts)

# -------------------------------------------------------------
# 2. FEATURE ENGINEERING & HANDLING MISSING DATA
# -------------------------------------------------------------
print("\n" + "="*80)
print("2. FEATURE ENGINEERING & MISSING VALUE IMPUTATION")
print("="*80)

# Create a copy for feature engineering
df_fe = df.copy()

# A. Introduce artificial missing data demo if dataset is completely clean,
#    to showcase robust real-world imputation techniques (Mean, Median, Mode)
np.random.seed(42)
demo_missing = df_fe.copy()
mask_area = np.random.rand(len(demo_missing)) < 0.05
mask_furnishing = np.random.rand(len(demo_missing)) < 0.05
demo_missing.loc[mask_area, 'area'] = np.nan
demo_missing.loc[mask_furnishing, 'furnishingstatus'] = np.nan

print(f"Demonstration: Created simulated missing values in copy:")
print(f" - Missing 'area' (numerical): {demo_missing['area'].isnull().sum()}")
print(f" - Missing 'furnishingstatus' (categorical): {demo_missing['furnishingstatus'].isnull().sum()}")

# Imputation demonstration
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

imputed_area = num_imputer.fit_transform(demo_missing[['area']])
imputed_furnishing = cat_imputer.fit_transform(demo_missing[['furnishingstatus']])
print("Imputation applied successfully: 0 missing values remain in demonstration copy.")

# B. Domain-Specific Feature Engineering
print("\nEngineering new domain-specific features:")
# 1. Total rooms
df_fe['total_rooms'] = df_fe['bedrooms'] + df_fe['bathrooms']

# 2. Area per bedroom (Living space density)
df_fe['area_per_bedroom'] = df_fe['area'] / df_fe['bedrooms']

# 3. Area per total room
df_fe['area_per_room'] = df_fe['area'] / df_fe['total_rooms']

# 4. Binary conversion for binary categorical features
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df_fe[col + '_code'] = (df_fe[col].str.lower() == 'yes').astype(int)

# 5. Amenities Score (Count of premium features)
df_fe['amenities_score'] = (
    df_fe['mainroad_code'] +
    df_fe['guestroom_code'] +
    df_fe['basement_code'] +
    df_fe['hotwaterheating_code'] +
    df_fe['airconditioning_code'] +
    df_fe['prefarea_code'] +
    (df_fe['parking'] > 0).astype(int)
)

# 6. Luxury Interaction feature: Area * Bathrooms * AirConditioning
df_fe['luxury_index'] = df_fe['area'] * df_fe['bathrooms'] * (df_fe['airconditioning_code'] + 1)

print("Created features: 'total_rooms', 'area_per_bedroom', 'area_per_room', 'amenities_score', 'luxury_index'")

# Define classification target: High-Value Property (Price >= Median Price)
price_median = df_fe['price'].median()
df_fe['high_value_target'] = (df_fe['price'] >= price_median).astype(int)
print(f"\nClassification Target: 'high_value_target' (Price >= ${price_median:,.0f})")
print(f"Class Distribution: \n{df_fe['high_value_target'].value_counts(normalize=True)}")

# -------------------------------------------------------------
# 3. FEATURE SCALING COMPARISON
# -------------------------------------------------------------
print("\n" + "="*80)
print("3. FEATURE SCALING (StandardScaler vs MinMaxScaler vs RobustScaler)")
print("="*80)

num_features_to_scale = ['area', 'area_per_bedroom', 'luxury_index']
scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_robust = RobustScaler()

scaled_std = scaler_std.fit_transform(df_fe[num_features_to_scale])
scaled_minmax = scaler_minmax.fit_transform(df_fe[num_features_to_scale])
scaled_robust = scaler_robust.fit_transform(df_fe[num_features_to_scale])

# Plotting Feature Scaling Comparison
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
scalers_dict = {
    'Original (Unscaled)': df_fe[num_features_to_scale].values,
    'StandardScaler (Z-Score)': scaled_std,
    'MinMaxScaler ([0, 1])': scaled_minmax,
    'RobustScaler (IQR)': scaled_robust
}

for row_idx, feat in enumerate(num_features_to_scale):
    for col_idx, (s_name, s_data) in enumerate(scalers_dict.items()):
        ax = axes[row_idx, col_idx]
        sns.histplot(s_data[:, row_idx], kde=True, ax=ax, color='#1f77b4' if col_idx==0 else '#2ca02c')
        ax.set_title(f"{feat} - {s_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

plt.tight_layout()
scaling_plot_path = os.path.join(PLOTS_DIR, "scaling_comparison.png")
plt.savefig(scaling_plot_path, dpi=300)
plt.close()
print(f"Saved scaling comparison plot to: {scaling_plot_path}")

# -------------------------------------------------------------
# 4. PREPARATION FOR MODEL EVALUATION (TRAIN/TEST SPLIT)
# -------------------------------------------------------------
print("\n" + "="*80)
print("4. TRAIN / TEST SPLIT & PIPELINE SETUP")
print("="*80)

feature_cols = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'total_rooms', 'area_per_bedroom', 'area_per_room', 'amenities_score', 'luxury_index',
    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
    'furnishingstatus'
]

X = df_fe[feature_cols]
y = df_fe['high_value_target']

numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                  'total_rooms', 'area_per_bedroom', 'area_per_room', 'amenities_score', 'luxury_index']
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Stratified Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print(f"Target distribution in train: {np.bincount(y_train)}")
print(f"Target distribution in test:  {np.bincount(y_test)}")

# Preprocessing Pipeline (Imputation + Encoding + Scaling)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# -------------------------------------------------------------
# 5. CROSS-VALIDATION ACROSS MULTIPLE MODELS
# -------------------------------------------------------------
print("\n" + "="*80)
print("5. CROSS-VALIDATION EVALUATION (5-Fold Stratified CV)")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
    'Support Vector Classifier': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7)
}

cv_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results_summary = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_validate(pipe, X_train, y_train, cv=skf, scoring=cv_scoring, return_train_score=False)
    
    cv_results_summary.append({
        'Model': name,
        'Accuracy (Mean)': scores['test_accuracy'].mean(),
        'Accuracy (Std)': scores['test_accuracy'].std(),
        'Precision (Mean)': scores['test_precision'].mean(),
        'Recall (Mean)': scores['test_recall'].mean(),
        'F1-Score (Mean)': scores['test_f1'].mean(),
        'ROC-AUC (Mean)': scores['test_roc_auc'].mean(),
        'ROC-AUC (Std)': scores['test_roc_auc'].std(),
    })

cv_df = pd.DataFrame(cv_results_summary)
print(cv_df.to_string(index=False))

# Plot CV Scores Comparison
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['Accuracy (Mean)', 'Precision (Mean)', 'Recall (Mean)', 'F1-Score (Mean)', 'ROC-AUC (Mean)']
cv_df_plot = cv_df.set_index('Model')[metrics_to_plot]
cv_df_plot.plot(kind='bar', ax=ax, colormap='viridis', width=0.8, edgecolor='black')
ax.set_title("5-Fold Cross-Validation Metrics Comparison Across Models", fontsize=14, fontweight='bold')
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.6, 1.0)
ax.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.tight_layout()
cv_plot_path = os.path.join(PLOTS_DIR, "cv_scores_comparison.png")
plt.savefig(cv_plot_path, dpi=300)
plt.close()
print(f"Saved CV scores plot to: {cv_plot_path}")

# -------------------------------------------------------------
# 6. MODEL EVALUATION ON TEST SET (CONFUSION MATRIX, P-R, ROC-AUC)
# -------------------------------------------------------------
print("\n" + "="*80)
print("6. TEST SET EVALUATION: CONFUSION MATRIX, PRECISION/RECALL/F1, ROC-AUC")
print("="*80)

fig_cm, axes_cm = plt.subplots(2, 3, figsize=(16, 10))
axes_cm = axes_cm.flatten()

fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
fig_pr, ax_pr = plt.subplots(figsize=(9, 7))

test_evaluation_summary = []

for idx, (name, model) in enumerate(models.items()):
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    
    test_evaluation_summary.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Avg Precision (PR-AUC)': ap
    })
    
    print(f"\n--- {name.upper()} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Standard (<Median)', 'High-Value (>=Median)']))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Standard', 'High-Value'])
    disp.plot(ax=axes_cm[idx], cmap='Blues', colorbar=False)
    axes_cm[idx].set_title(f"{name}\n(F1: {f1:.3f} | Acc: {acc:.3f})", fontsize=11, fontweight='bold')
    axes_cm[idx].grid(False)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ax_pr.plot(recall_vals, precision_vals, lw=2, label=f'{name} (AP = {ap:.3f})')

# Remove empty subplot in CM grid if any
if len(models) < len(axes_cm):
    fig_cm.delaxes(axes_cm[-1])

fig_cm.suptitle("Confusion Matrices on Test Set (Housing Dataset)", fontsize=15, fontweight='bold')
fig_cm.tight_layout()
cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
fig_cm.savefig(cm_plot_path, dpi=300)
plt.close(fig_cm)
print(f"Saved Confusion Matrices plot to: {cm_plot_path}")

# Finalize ROC Curve Plot
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax_roc.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
ax_roc.legend(loc="lower right", fontsize=10)
fig_roc.tight_layout()
roc_plot_path = os.path.join(PLOTS_DIR, "roc_auc_curves.png")
fig_roc.savefig(roc_plot_path, dpi=300)
plt.close(fig_roc)
print(f"Saved ROC-AUC curves plot to: {roc_plot_path}")

# Finalize Precision-Recall Curve Plot
no_skill = len(y_test[y_test == 1]) / len(y_test)
ax_pr.plot([0, 1], [no_skill, no_skill], 'k--', lw=1.5, label=f'Baseline ({no_skill:.2f})')
ax_pr.set_xlim([0.0, 1.0])
ax_pr.set_ylim([0.0, 1.05])
ax_pr.set_xlabel('Recall', fontsize=12)
ax_pr.set_ylabel('Precision', fontsize=12)
ax_pr.set_title('Precision-Recall (PR) Curves', fontsize=14, fontweight='bold')
ax_pr.legend(loc="lower left", fontsize=10)
fig_pr.tight_layout()
pr_plot_path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
fig_pr.savefig(pr_plot_path, dpi=300)
plt.close(fig_pr)
print(f"Saved Precision-Recall curves plot to: {pr_plot_path}")

# Print summary table
test_df = pd.DataFrame(test_evaluation_summary)
print("\n" + "="*80)
print("FINAL TEST SET PERFORMANCE SUMMARY")
print("="*80)
print(test_df.to_string(index=False))

# -------------------------------------------------------------
# 7. FEATURE IMPORTANCE ANALYSIS
# -------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
pipe_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])
pipe_rf.fit(X_train, y_train)

# Extract feature names after one-hot encoding
cat_encoder = pipe_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_feature_names = numerical_cols + cat_feature_names

importances = pipe_rf.named_steps['classifier'].feature_importances_
feat_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', hue='Feature', palette='mako', legend=False)
plt.title('Top 10 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Gini Feature Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
fi_plot_path = os.path.join(PLOTS_DIR, "feature_importance.png")
plt.savefig(fi_plot_path, dpi=300)
plt.close()
print(f"Saved Feature Importance plot to: {fi_plot_path}")

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)
