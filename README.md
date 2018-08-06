# Predicting House Prices

[Presentation slides](slides_housing_price_capstone.pdf)<br>
[Jupyter notebook](unit_03_capstone_final_notebook.ipynb)

**Toolkit**
- Python 3, NumPy, Pandas, Matplotlib, Seaborn, SciPy, SKLearn, XGBoost

**Topics**
- EDA, data cleaning, preprocessing, visualization
- Feature selection, engineering, dimensionality reduction
- Regularized linear regression, random forest, gradient boosting

# 1. Introduction
- Project goal: predict residential house prices using descriptive data about properties sold
- Dataset: Ames, Iowa Housing Market Data prepared by Dean De Cock
  - 23 nominal variables
  - 23 ordinal variables
  - 14 discrete variables
  - 20 continuous variables

# 2. Data Preprocessing
### Summary of Preprocessing
- Outliers mentioned in dataset documentation excluded
- Missingness: imputed 0 or column mode/mean
- Target variable distribution:
  - Initial skew = 0.1566
  - Log transformed to satisfy linear regression assumption of normally distributed residuals
  - Post log skew = 0.065
- Features created: yrmo_sold, age_at_sale
- Collinear variables: either performed PCA, scaled & average, or dropped
- Large amounts of ordinal variables to address
- Ordinal split into four subsets for comparison
  - Ordinal categorical
  - Ordinal continuous
  - Ordinal mixed cat/cont
  - Ordinal untransformed from original type

# 3. Modeling
### Models & Evaluation
- Linear regression: L1, L2, elastic net regularized linear regression
- Ensemble models: random forest, gradient boosted decision trees
- Evaluation metric: RMSE
### Algorithm Performance
-Lasso and elastic net regularization consistently produced similarly low RMSE scores
- L1 regularization:
  - **RMSE: 0.11052**
  - alpha = 0.0005
- Elastic net regularization:
  - **RMSE: 0.11056**
  - alpha = 0.0005
  - L1 ratio = 0.95 (near full weight given to L1, this parameter never took a value of 1 in cv tuning)
- Gradient boosted decision trees (XGBoost): only model to perform best on mixed ordinal data subset
  - **RMSE: 0.11994**
  - Parameters tuned via gridsearch
  - booster = gbtree
  - learning rate = 0.1
  - max depth = 5
  - subsample = 0.5
