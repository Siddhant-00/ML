# Linear Regression on Housing Dataset

## Overview
This project predicts housing prices using a Linear Regression model on the Boston Housing Dataset. The workflow includes preprocessing, model training, evaluation, and visualization.

## Dataset
- **Features (X):** Factors influencing house prices, such as crime rate and average number of rooms.
- **Target (y):** `MEDV` (Median value of owner-occupied homes in $1000s).

## Prerequisites
Install the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Workflow
1. **Import Libraries:**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import r2_score
   ```

2. **Load Dataset:**
   ```python
   dataset = pd.read_csv(r"C:\Users\HP\Desktop\housing.csv")
   X = dataset.drop(columns=['MEDV'])
   y = dataset['MEDV']
   ```

3. **Split Data:**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
   ```

4. **Standardize Features:**
   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

5. **Train Model:**
   ```python
   regression = LinearRegression()
   regression.fit(X_train, y_train)
   ```

6. **Cross-Validate:**
   ```python
   mse = cross_val_score(regression, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
   print(np.mean(mse))
   ```

7. **Predict and Evaluate:**
   ```python
   reg_pred = regression.predict(X_test)
   score = r2_score(y_test, reg_pred)
   print(score)
   ```

8. **Visualize Residuals:**
   ```python
   sns.displot(reg_pred - y_test, kind='kde')
   plt.show()
   ```

## Output
- **Residual Plot:** Visualizes prediction errors.
- **R-squared Score:** Indicates model performance.

## Usage
1. Replace the dataset file path in `pd.read_csv`.
2. Run the script in a Python environment (e.g., Jupyter Notebook).

## License
This project is licensed under the MIT License.

