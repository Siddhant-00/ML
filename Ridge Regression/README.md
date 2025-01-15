# Ridge Regression with Grid Search for Hyperparameter Tuning

## Overview
This project implements a Ridge Regression model to predict housing prices using the Boston Housing dataset. The workflow includes data preprocessing, feature scaling, hyperparameter tuning using GridSearchCV, and model evaluation. The project is coded in Python and leverages popular libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

## Dataset
The dataset used in this project is the Boston Housing dataset (`housing.csv`). It contains the following columns:
- **Independent Features (X)**: All columns except `MEDV`.
- **Dependent Feature (y)**: The `MEDV` column, representing the Median Value of owner-occupied homes in $1000's.

## Prerequisites
To run this project, you need to have Python installed along with the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Installation
1. Clone this repository or download the code files.
2. Install the required libraries by running:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

## Workflow
1. **Load the Dataset**
   Load the dataset from a CSV file using Pandas.

2. **Split the Data**
   Split the data into independent features (`X`) and the dependent feature (`y`). Use `train_test_split` from `sklearn` to divide the data into training and testing sets (70-30 split).

3. **Standardize Features**
   Standardize the features using `StandardScaler` to improve model performance.

4. **Implement Ridge Regression**
   Use Ridge Regression for the prediction task. Perform hyperparameter tuning with `GridSearchCV` to find the optimal value of the regularization parameter `alpha`.

5. **Evaluate the Model**
   - Display the best hyperparameters and the corresponding negative mean squared error.
   - Predict values on the test set and visualize the residual distribution using a Kernel Density Estimation (KDE) plot.
   - Calculate the R² score to assess model performance.

## Key Code Snippets
### Load and Split Data
```python
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("path/to/housing.csv")
X = dataset.drop(columns=['MEDV'])
y = dataset['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```

### Standardize Features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Ridge Regression with Grid Search
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge_regressor = Ridge()
parameters = {'alpha': [1, 2, 5, 10, 20, 40, 50, 60, 70, 80, 90]}
ridgecv = GridSearchCV(ridge_regressor, parameters, scoring='neg_mean_squared_error', cv=5)
ridgecv.fit(X_train, y_train)

print("Best Parameters:", ridgecv.best_params_)
print("Best Score:", ridgecv.best_score_)
```

### Evaluate the Model
```python
import seaborn as sns
from sklearn.metrics import r2_score

ridge_pred = ridgecv.predict(X_test)
sns.displot(ridge_pred - y_test, kind='kde')

score = r2_score(ridge_pred, y_test)
print("R² Score:", score)
```

## Results
- **Best Parameters**: The optimal value of `alpha` obtained through GridSearchCV.
- **Best Score**: The negative mean squared error for the optimal model.
- **R² Score**: The coefficient of determination for the model on the test data.

## Visualization
The KDE plot shows the distribution of residuals (differences between predicted and actual values). Ideally, the residuals should be centered around zero, indicating good model performance.

## Future Improvements
- Explore additional hyperparameters for Ridge Regression.
- Compare Ridge Regression with other regression models like Lasso and ElasticNet.
- Perform feature selection to identify the most impactful features.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Scikit-learn documentation for Ridge Regression and GridSearchCV.
- Boston Housing dataset for providing the data used in this project.

## Author
[Siddhant Jain]

