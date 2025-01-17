# README: Lasso Regression with Grid Search CV

This document provides an overview and step-by-step guide for the Python script that performs Lasso Regression on a housing dataset. The script includes data preparation, model training, hyperparameter tuning, and evaluation.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python (>=3.6)
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

Install any missing libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used in this script should be a CSV file containing housing data with the target variable labeled as `MEDV` (Median Value of Owner-Occupied Homes). Place the dataset file in the specified location on your machine and update the file path accordingly.

## Script Workflow

### 1. Import Libraries
The required libraries for data manipulation, visualization, and machine learning are imported:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import seaborn as sns
```

### 2. Load Dataset
The dataset is loaded using pandas:
```python
dataset = pd.read_csv(r"C:\Users\HP\Desktop\housing.csv")
```

### 3. Split Data into Features and Target
Independent features (`X`) and the dependent target variable (`y`) are separated:
```python
X = dataset.drop(columns=['MEDV'])
y = dataset['MEDV']
```

### 4. Split Data into Training and Testing Sets
The data is split into training and testing subsets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```

### 5. Standardize the Data
Feature scaling is applied using `StandardScaler`:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. Lasso Regression Model
A Lasso Regression model is initialized and tuned using GridSearchCV:
```python
lasso_regressor = Lasso()
parameters = {'alpha': [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]}
lassocv = GridSearchCV(lasso_regressor, parameters, scoring='neg_mean_squared_error', cv=5)
lassocv.fit(X_train, y_train)
```

### 7. Optimal Hyperparameters
The best alpha value and corresponding score are displayed:
```python
print(lassocv.best_params_)
print(lassocv.best_score_)
```

### 8. Predictions
Predictions are made on the test dataset:
```python
lasso_pred = lassocv.predict(X_test)
```

### 9. Visualization
The distribution of residuals is visualized using Seaborn:
```python
sns.displot(lasso_pred - y_test, kind='kde')
```

### 10. Model Evaluation
The R-squared score is computed to evaluate the model:
```python
score = r2_score(lasso_pred, y_test)
print(score)
```

## Output
1. Best alpha value (`lassocv.best_params_`)
2. Best negative mean squared error score (`lassocv.best_score_`)
3. Visualization of residuals
4. R-squared score

## Notes
- Ensure the dataset is correctly formatted and the path is accurate.
- Modify the hyperparameter grid (`parameters`) as needed for better tuning.



