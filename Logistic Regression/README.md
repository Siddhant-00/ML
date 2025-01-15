# Logistic Regression for Customer Churn Prediction

## Overview
This project uses logistic regression to predict customer churn based on two features: `tenure` and `MonthlyCharges`. The dataset is pre-processed and analyzed to evaluate the model's performance using metrics such as accuracy and confusion matrix.

## Dataset
The dataset used is `Telco-Customer-Churn.csv`. The key columns used in the model are:
- **tenure**: Number of months the customer has stayed with the company.
- **MonthlyCharges**: The monthly charges incurred by the customer.
- **Churn**: Target variable indicating if a customer has churned (`Yes`) or not (`No`).

## Steps

### 1. Data Loading and Preparation
- Load the dataset using `pandas`.
- Inspect the number of rows and columns.
- Convert the `Churn` column from string (`Yes`/`No`) to integer values (`1` for Yes, `0` for No).
- Create feature set `X` containing `tenure` and `MonthlyCharges`, and target variable `y` containing `class` (binary churn indicator).

### 2. Exploratory Data Analysis (EDA)
- Boxplots are created to observe the distribution of `MonthlyCharges` and `tenure` across churn classes.
  - Insights:
    - A visible difference in tenure distribution between churn classes.
    - Slight differences in monthly charges between churn classes.

### 3. Train-Test Split
- Split the data into training and testing sets using an 80-20 split.
- Check the distribution of the target variable in both sets to ensure a good balance.

### 4. Logistic Regression Model
- Fit a logistic regression model using `LogisticRegression` from `sklearn`.
- Extract the model coefficients and intercept to understand the relationship between features and target.

### 5. Model Evaluation
#### Predicted Probabilities
- Obtain predicted probabilities for both training and testing sets.

#### Predicted Classes
- Obtain class predictions (0 or 1) for training and testing sets.

#### Accuracy Scores
- Calculate and print accuracy scores for training and testing datasets.

#### Confusion Matrix
- Generate and visualize the confusion matrix using `seaborn` for both training and testing sets.

### 6. Cross-Validated Logistic Regression
- Use `LogisticRegressionCV` and `cross_validate` to implement cross-validated logistic regression.
- Evaluate the performance across 5 folds using accuracy as the scoring metric.
- Print model coefficients for each fold to observe variations.

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

Install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run
1. Place the `Telco-Customer-Churn.csv` file in the desired location.
2. Update the file path in the `pd.read_csv()` function.
3. Execute the script in a Python IDE or Jupyter Notebook.

## Outputs
- Boxplots for `MonthlyCharges` and `tenure`.
- Confusion matrices for training and testing sets.
- Accuracy scores for training and testing sets.
- Coefficients and intercept of the logistic regression model.
- Cross-validation scores and estimators.

## Insights
- The `tenure` feature shows a clear distinction between churned and retained customers, making it a strong predictor.
- The model achieves a good balance between training and testing accuracy.
- Cross-validation provides robustness by validating the model's performance across multiple folds.

## Next Steps
- Include additional features to improve model accuracy.
- Experiment with advanced models like decision trees or ensemble methods.
- Perform feature scaling to evaluate its impact on model performance.

## Author
[Siddhant Jain]



