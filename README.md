# Machine Learning Projects

A collection of machine learning notebooks where I apply and test what I've learned. Each project is a self-contained notebook that goes from raw data to a trained and evaluated model.

If you spot a mistake or something that could be improved, please open an issue or tell me. Feedback is welcome.

## Projects

| # | Project | Model | Notebook |
|---|---------|-------|----------|
| 1 | NYC house price prediction: feature selection + Box-Cox | XGBoost | [Notebook](<./NYC HOUSE PRICE PREDICTION WITH FEATURE SELECTION AND BOX COX TRANSFOR,.ipynb>) |
| 2 | NYC house price prediction: dense neural network | Keras / TensorFlow | [Notebook](<./NYC HOUSING PRICE PREDICTON DENSE NEURAL NETWORKS.ipynb>) |

Both projects use the same dataset, `nyc_housing_base.csv`, which lives in the [`data/`](./data) folder. The target is `sale_price`.

---

## 1. NYC House Price Prediction: XGBoost with Feature Selection and Box-Cox

**Goal:** predict NYC home sale prices with a gradient boosted tree model, and see how much feature selection and a target transform help.

**Pipeline**

1. **Cleaning:** drop rows with missing values and remove price outliers using the 1.5 × IQR rule.
2. **Stratified split:** bin `sale_price` into 10 quantile buckets (`pd.qcut`) and use them to stratify an 80 / 10 / 10 train / validation / test split, so each split has a similar price distribution.
3. **Encoding:** one-hot encode the categorical columns with `pd.get_dummies`.
4. **Feature selection:** keep the top 100 features using `SelectKBest` with `f_regression`, fitted on the training set only.
5. **Target transform:** apply a Box-Cox `PowerTransformer` to `sale_price` to reduce skew (fitted on train, then applied to validation and test).
6. **Model:** `XGBRegressor(n_estimators=1500, learning_rate=0.006, max_depth=7)`.
7. **Evaluation:** predictions are inverse-transformed back to dollars and scored with mean absolute error (MAE).

**Results**

| Split | MAE |
|-------|-----|
| Train | $197,515 |
| Validation | $218,236 |

For context, the mean sale price after cleaning is about $810,329, so validation MAE is roughly 27% of the mean price.

---

## 2. NYC House Price Prediction: Dense Neural Network

**Goal:** model the same problem with a fully connected neural network and compare it with the tree-based approach.

**Pipeline**

1. **Cleaning:** drop `zip_code`, remove price outliers with the 1.5 × IQR rule, and drop missing values.
2. **Split:** sequential 80 / 10 / 10 train / validation / test split (no shuffling).
3. **Encoding:** one-hot encode categorical columns. Validation and test columns are reindexed to match the training columns, giving 115 input features.
4. **Scaling:** standardize features and target using the training mean and standard deviation, and convert to `float32`.
5. **Model:** a `Sequential` network with layers of 256 (L2 regularization, 0.01), 128, 64, 32, 16 and 1 units. The hidden layers use ReLU. Trained with Adam and MAE loss for 64 epochs at batch size 32.
6. **Evaluation:** predictions are un-scaled back to dollars and scored with MAE. A plot compares validation predictions against actual prices.

**Results**

| Split | MAE |
|-------|-----|
| Train | $281,551 |
| Validation | $289,982 |

The train and validation errors are close, which suggests the network is not overfitting. It is, however, noticeably less accurate than the XGBoost model above.

> **Note:** the two notebooks split the data differently (stratified random vs. sequential) and handle outliers and missing values in a slightly different order, so the comparison between them is indicative rather than strictly like-for-like. The test sets are created in both notebooks but not yet scored.

---

## Repository Structure

```
MACHINE-LEARNING-PROJECTS/
├── data/
│   └── nyc_housing_base.csv
├── NYC HOUSE PRICE PREDICTION WITH FEATURE SELECTION AND BOX COX TRANSFOR,.ipynb
├── NYC HOUSING PRICE PREDICTON DENSE NEURAL NETWORKS.ipynb
└── README.md
```

## Getting Started

**Requirements:** Python 3.12 and the following packages:

```
pandas
numpy
scikit-learn
xgboost
tensorflow
matplotlib
jupyter
```

**Run**

```bash
git clone https://github.com/purush-1/MACHINE-LEARNING-PROJECTS.git
cd MACHINE-LEARNING-PROJECTS
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib jupyter
jupyter notebook
```

The notebooks currently load the CSV from a local path on my machine. Before running, change the `pd.read_csv(...)` line to point to the file in this repo:

```python
df = pd.read_csv('data/nyc_housing_base.csv')
```

## Ideas for Improvement

- Score the held-out test set for both models.
- Tune the XGBoost and neural network hyperparameters (e.g. with Optuna or cross-validation).
- Use the same split strategy in both notebooks for a fair comparison.
- Try early stopping and a validation curve for the neural network.

## Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · TensorFlow / Keras · Matplotlib
