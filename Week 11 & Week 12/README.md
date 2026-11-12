# 🌐 Week 11 & Week 12: Unsupervised Learning & Dimensionality Reduction (PCA & t-SNE)

An in-depth, mathematically grounded laboratory on **Dimensionality Reduction** and **Unsupervised Manifold Learning**, featuring **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)** applied to global socioeconomic development data.

---

## 📌 Project Overview

High-dimensional data in machine learning introduces the **Curse of Dimensionality**: exponential volume expansion, distance metric degradation, high computational overhead, and extreme multicollinearity. This module explores both **linear** and **non-linear** dimensionality reduction techniques to uncover latent structures, visualize complex relationships, and profile countries for humanitarian and economic development aid.

### 🔬 Core Exploration Tracks

```
                                  ┌─────────────────────────────────────────┐
                                  │   High-Dimensional Socioeconomic Data   │
                                  │      (167 Countries × 9 Indicators)     │
                                  └────────────────────┬────────────────────┘
                                                       │
                                        StandardScaler Feature Centering
                                                       │
                        ┌──────────────────────────────┴──────────────────────────────┐
                        ▼                                                             ▼
         ┌─────────────────────────────┐                               ┌─────────────────────────────┐
         │     Linear Reduction        │                               │     Non-Linear Manifold     │
         │  Principal Component (PCA)  │                               │       Embedding (t-SNE)     │
         ├─────────────────────────────┤                               ├─────────────────────────────┤
         │ • Covariance Eigen-decomp   │                               │ • High-D Gaussian Affinities│
         │ • Variance Maximization     │                               │ • Student-t Low-D Mapping   │
         │ • Scree & Kaiser Selection  │                               │ • Resolves Crowding Problem │
         │ • Loadings & Biplot         │                               │ • Perplexity Spectrum Study │
         │ • KMeans in Latent Space    │                               │ • Non-linear Cluster Maps   │
         │ • Exact Data Reconstruction │                               │ • Hybrid: PCA ➔ t-SNE       │
         └─────────────────────────────┘                               └─────────────────────────────┘
```

---

## 🗂️ Repository & Directory Architecture

```
Week 11 & Week 12/
├── Country-data.csv        # Global socioeconomic indicators dataset (167 nations)
├── data-dictionary.csv     # Official field definitions & measurement units
├── PCA.ipynb               # Linear reduction, mathematical proofs, biplots & clustering
├── t-SNE.ipynb             # Non-linear manifold embedding, perplexity dynamics & hybrid pipeline
└── README.md               # Comprehensive documentation and technical guide
```

---

## 📊 Dataset Description

The dataset comprises **167 countries** evaluated across **9 socioeconomic and health metrics** sourced for global development profiling:

| Feature | Description | Unit / Scale |
| :--- | :--- | :--- |
| `country` | Name of the country | Categorical identifier |
| `child_mort` | Under-5 child mortality rate | Deaths per 1,000 live births |
| `exports` | Exports of goods and services per capita | % of GDP per capita |
| `health` | Total public & private health expenditure | % of GDP per capita |
| `imports` | Imports of goods and services per capita | % of GDP per capita |
| `income` | Net income per person | USD (purchasing power adjusted) |
| `inflation` | Annual GDP growth rate of inflation | Annual percentage (%) |
| `life_expec` | Average life expectancy at birth | Years |
| `total_fer` | Total fertility rate | Average children born per woman |
| `gdpp` | Gross Domestic Product per capita | USD per person |

---

## 📐 Part 1: Principal Component Analysis (`PCA.ipynb`)

### 1. Mathematical Foundations
- **Objective Formulation**: Finds orthogonal axes $w_1, w_2, \dots, w_k \in \mathbb{R}^d$ maximizing the projected data variance:
  $$\max_{w_1} w_1^T \Sigma w_1 \quad \text{subject to} \quad w_1^T w_1 = 1$$
- **Lagrangian Optimization**: 
  $$\mathcal{L}(w_1, \lambda_1) = w_1^T \Sigma w_1 - \lambda_1 (w_1^T w_1 - 1) \implies \mathbf{\Sigma w_1 = \lambda_1 w_1}$$
  The optimal projection directions correspond directly to the **eigenvectors** of covariance matrix $\Sigma$, with captured variance equal to **eigenvalues** $\lambda_i$.
- **Reconstruction Duality**: Maximizing projected variance is mathematically equivalent to minimizing the mean squared reconstruction error:
  $$\min_{W_k} \frac{1}{n}\sum_{i=1}^n \|x_i - W_k W_k^T x_i\|^2$$

### 2. Key Analytical Stages
1. **Multicollinearity & Scale Disparity**: Proves why raw PCA fails when features operate on drastically different scales (e.g., `gdpp` spanning tens of thousands vs. `total_fer` spanning 1–7) and standardizes data using `StandardScaler`.
2. **Eigen-Decomposition Verification**: Verifies scikit-learn's singular value decomposition against manual covariance eigen-decomposition via `np.linalg.eigh`.
3. **Component Selection**:
   - **Scree Plot & Elbow Rule**: Identifies the rate of variance decline per added component.
   - **Kaiser's Criterion**: Retains components with eigenvalues $\lambda_i > 1$.
   - **Cumulative Explained Variance**: Evaluates the threshold ($>85\%$) required for low-loss representation.
4. **Feature Loadings & Economic Interpretation**:
   - **PC1 (Development & Quality of Life Index)**: Strong positive loadings on `gdpp`, `income`, `life_expec`; strong negative loadings on `child_mort` and `total_fer`.
   - **PC2 (International Trade Openness)**: Dominated by `exports` and `imports`.
   - **PC3 (Macroeconomic Stability)**: Driven primarily by `inflation`.
5. **Visualizations**:
   - **2D Country Mapping**: Spatial segmentation of under-developed, developing, and developed nations.
   - **PCA Biplot**: Superimposed loading vectors revealing variable directions and correlations alongside sample points.
   - **3D Interactive/Scatter Projection**: Capturing over $75\%$ of total variance across three axes.
6. **Unsupervised Profiling**: KMeans clustering applied in the lower-dimensional PCA subspace to isolate countries requiring immediate humanitarian assistance.
7. **Reconstruction Fidelity**: Measures mean squared reconstruction error across $k \in [1, 9]$ components.

---

## 🧠 Part 2: t-Distributed Stochastic Neighbor Embedding (`t-SNE.ipynb`)

### 1. Why Non-Linear Manifold Learning?
Linear PCA projects along hyperplanes, causing folded manifolds and non-linear boundaries to collapse. **t-SNE** prioritizes **local preservation of neighborhoods** over global Euclidean distances.

### 2. Mathematical Architecture
- **High-Dimensional Gaussian Affinities**:
  $$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
  where bandwidth $\sigma_i$ is adaptively solved via binary search to match a specified **Perplexity**:
  $$\text{Perplexity}(P_i) = 2^{H(P_i)}$$
- **Low-Dimensional Student-t Distribution (Resolving the Crowding Problem)**:
  $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$
  The heavy polynomial tails ($q \propto 1/d^2$) allow moderately distant points to spread out in 2D space without artificially collapsing into the origin.
- **Kullback-Leibler Divergence Minimization**:
  $$\mathcal{L}_{\text{KL}} = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$
  Optimized via gradient descent with early exaggeration and momentum.

### 3. Key Experimental Insights
1. **Perplexity Spectrum Sweep ($5, 15, 30, 50, 100$)**:
   - *Low Perplexity (5)*: Produces fragmented micro-clusters and artificial grouping artifacts.
   - *Optimal Perplexity (30–50)*: Striking balance between local neighborhood structure and continental-scale macroeconomic clusters.
   - *High Perplexity (100)*: Oversmooths local clusters, tending toward a uniform spread.
2. **Convergence Dynamics**: Explores the impact of learning rate, early exaggeration, and iteration thresholds on final embedding stability.
3. **t-SNE Interpretation Caveats**:
   - Cluster density in t-SNE does **not** equal high-dimensional variance or density.
   - Relative distances between separated clusters cannot be interpreted quantitatively.
   - Embeddings are stochastic; random seeds must be fixed for reproducibility.
4. **Industry Best Practice: PCA + t-SNE Hybrid**:
   - Initial reduction via PCA to 5 components (filtering noise and eliminating collinearity).
   - Subsequent t-SNE mapping to 2D for optimized convergence speed and cleaner manifold visualization.

---

## ⚔️ Comparison: PCA vs. t-SNE

| Attribute | Principal Component Analysis (PCA) | t-Distributed Stochastic Neighbor Embedding (t-SNE) |
| :--- | :--- | :--- |
| **Model Type** | Linear, parametric projection | Non-linear, non-parametric manifold learning |
| **Optimization** | Convex (Exact eigen-decomposition / SVD) | Non-convex (Gradient descent on KL divergence) |
| **Distance Preservation** | Global pairwise distances & variance | Local neighborhood affinities |
| **Invertibility** | Fully invertible ($\hat{X} = Z W^T$) | Non-invertible (Visualization / mapping only) |
| **Determinism** | Fully deterministic | Stochastic (depends on random initialization) |
| **New Data Projection** | Supported via `transform(X_new)` | Not supported directly (must refit or approximate) |
| **Computation Speed** | Extremely fast ($\mathcal{O}(d^2 n + d^3)$) | Computationally intensive ($\mathcal{O}(n^2)$) |
| **Primary Use Cases** | Feature reduction, noise filtering, preprocessing | Exploratory visualization, cluster topology discovery |

---

## 💻 Setup & Execution

### 1. Prerequisites
Ensure Python 3.8+ is installed. Clone or navigate to the repository:

```bash
git clone https://github.com/SiddhantBahekar/HACK-O-WEEK-5th-Sem.git
cd "HACK-O-WEEK-5th-Sem/Week 11 & Week 12"
```

### 2. Environment Setup
Create and activate an isolated virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install the required analytical libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Running the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
- Open `PCA.ipynb` to step through the linear decomposition, Scree plots, loadings, and clustering.
- Open `t-SNE.ipynb` to explore the non-linear manifold embeddings and perplexity experiments.

---

## 🚀 Git Commands to Commit and Push

Follow these steps from your terminal to stage, commit, and push this folder to your GitHub repository:

```bash
# 1. Navigate to the root directory of the repository
cd "d:\Symbiosis\SEM 5\PBL Hack O Week"

# 2. Check current status
git status

# 3. Stage the Week 11 & Week 12 files
git add "Week 11 & Week 12"

# 4. Commit the changes
git commit -m "feat(week-11-12): add comprehensive PCA and t-SNE notebooks, dataset, and documentation"

# 5. Push to your GitHub repository
git push origin main
```

---

## 👥 Academic & Course Context
- **Course**: Problem Based Learning (PBL) Hack O Week — Semester 5
- **Domain**: Unsupervised Machine Learning, Dimensionality Reduction & Manifold Learning
- **Author**: Siddhant Bahekar
