# MACHINE-LEARNING-PROJECTS

A collection of Machine Learning projects focused on applying preprocessing, feature engineering, feature selection, statistical transformations, and supervised learning models to real-world datasets. Each project documents the modeling approach, evaluation, and experiments performed during development.

## Repository Structure

| Notebook | Dataset | Approach |
|---|---|---|
| `NYC HOUSE PRICE PREDICTION WITH FEATURE SELECTION AND BOX COX TRANSFOR,.ipynb` | NYC housing sales data | Feature selection with `SelectKBest` + Box-Cox target transformation + XGBoost regression |

## Libraries Used

| Notebook | Libraries |
|---|---|
| NYC House Price Prediction | `pandas`, `numpy`, `matplotlib`, `xgboost` (`XGBRegressor`), `sklearn.metrics`, `sklearn.feature_selection`, `sklearn.preprocessing`, `sklearn.model_selection` |

## Model Results

### NYC House Price Prediction

| Metric | Result |
|---|---:|
| Train MAE | 197,515.44 |
| Validation MAE | 218,236.03 |
| Mean Sale Price | 810,329.37 |

The model's validation MAE is higher than its training MAE, indicating a generalization gap that can be investigated further through hyperparameter tuning, validation strategy, feature engineering, and model analysis.

## NYC House Price Prediction

**Dataset:** NYC housing sales data.

**Goal:** predict house sale prices using available housing features.

### Preprocessing

The dataset is processed before model training and includes:

- Price-based categorization for stratified splitting
- Train / validation / test split
- Feature selection
- Target transformation

The data is split into:

```text
Training:   80%
Validation: 10%
Test:       10%
```

The splits use stratification based on the created `Price_cat` variable.

### Feature Selection

`SelectKBest` with `f_regression` is used to select the top 100 features:

```python
selector = SelectKBest(score_func=f_regression, k=100)
```

The selector is fitted using the training data and then applied to the training, validation, and test sets.

### Target Transformation

The house-price target is transformed using a Box-Cox transformation:

```python
PowerTransformer(method="box-cox")
```

The transformation is fitted on the training target and then applied to the validation and test targets.

### Model

The project uses `XGBRegressor`:

```python
XGBRegressor(
    n_estimators=1500,
    learning_rate=0.006,
    max_depth=7
)
```

Predictions are transformed back to the original price scale before calculating MAE.

### Evaluation

The model is evaluated using **Mean Absolute Error (MAE)**.

```text
Train MAE:       197,515.44
Validation MAE:  218,236.03
```

Mean sale price:

```text
810,329.37
```

## How to Use

1. Install the required dependencies:

```bash
pip install pandas numpy matplotlib xgboost scikit-learn
```

2. Open the notebook in Jupyter:

```bash
jupyter notebook
```

3. Open:

```text
NYC HOUSE PRICE PREDICTION WITH FEATURE SELECTION AND BOX COX TRANSFOR,.ipynb
```

4. Run the notebook cells from top to bottom.

## Notes

This is a Machine Learning learning repository. Projects are being developed as new concepts and techniques are learned.

The focus is on understanding the complete modeling process rather than only obtaining a final score.

Future projects will expand into areas such as:

- Regression
- Classification
- Feature engineering
- Advanced model evaluation
- Time-series forecasting
- Deep learning
- Model deployment
- Production Machine Learning

Results and implementations will continue to evolve as the projects become more advanced.
