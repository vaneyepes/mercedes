## Starting the virtual enviroment

Install Environment from requirements.txt (in Mac OS):
Clone/download the project folder
Run:
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Workflow

### 1. Load Data

- `pd.read_csv("train.csv")`
- Use **train.csv** (with target `y`) for training and **test.csv** (without `y`) for final predictions.

### 2. Basic Data Review

- `df.info()`, `df.isnull().sum()`, `df.describe()`
- Check constant/low-variance features with `nunique()`.

### 3. Handle Missing Values

- None in this dataset → skip
- (General rule: median for numeric, mode for categorical).

### 4. Basic EDA

- Target distribution (`y`) → histogram, boxplot.
- Categorical distributions → countplot.
- Correlations with target → heatmap.
- Drop constant columns.

### 5. Define Features & Target

- `X = df.drop(['ID','y'], axis=1)`
- `y = df['y']`

### 6. Train/Test Split

- `train_test_split(X, y, test_size=0.2, random_state=42)`

---

## Linear Regression Workflow

### 7. Preprocessing

- `StandardScaler` for numeric features.
- `OneHotEncoder` for categorical features.

### 8. Baseline Models

- `LinearRegression()`
- Regularized models: `Ridge()`, `Lasso()`

### 9. Fit & Evaluate

- Metrics: **MSE, RMSE, R²**

### 10. Hyperparameter Tuning

- GridSearchCV for `alpha` (Ridge/Lasso).

---

## 🔹 Tree-Based Workflow

### 7. Preprocessing

- `OneHotEncoder` for categorical features.
- No scaling needed for numeric features.

### 8. Baseline Models

- `DecisionTreeRegressor()`
- `RandomForestRegressor()`

### 9. Fit & Evaluate

- Metrics: **MSE, RMSE, R²**

### 10. Hyperparameter Tuning

- GridSearchCV for `max_depth`, `min_samples_split`, `n_estimators`.

---

## 11. Compare Models

- Linear vs Regularized vs Tree-based
- Choose best based on RMSE / R²

