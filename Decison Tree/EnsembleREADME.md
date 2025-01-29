# Credit Card Fraud Detection

## Overview
This project aims to identify fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, so we use stratified sampling to ensure a balanced distribution in training and testing sets. We implement a Random Forest Classifier and use hyperparameter tuning to optimize model performance. The primary evaluation metric used is the ROC AUC score.

## Dataset
The dataset consists of credit card transactions with features representing transaction details. The last column of the dataset indicates whether the transaction is fraudulent (1) or non-fraudulent (0).

## Requirements
The following Python libraries are required to run the project:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install the dependencies using:
```sh
pip install pandas numpy matplotlib scikit-learn
```

## Steps

### 1. Load and Inspect the Data
- Load the dataset using pandas.
- Check the shape and distribution of the target variable.

### 2. Data Preprocessing
- Separate independent (X) and dependent (Y) variables.
- Perform stratified sampling to maintain the same distribution in training and testing sets.

### 3. Model Training
- Train a Random Forest Classifier.
- Perform hyperparameter tuning using GridSearchCV to find the best model parameters.

### 4. Model Evaluation
- Predict on training and test sets.
- Calculate accuracy and ROC AUC score.
- Generate confusion matrices to evaluate model performance.

### 5. Feature Importance
- Extract feature importances from the trained model.
- Visualize feature importance using a horizontal bar chart.

## Usage
Run the script to train the model and evaluate performance:
```sh
python fraud_detection.py
```
Ensure that the dataset is placed in the correct path before execution.

## Results
- The model is evaluated using the ROC AUC score as accuracy is not a suitable metric due to class imbalance.
- The confusion matrix provides insights into true positives, false positives, true negatives, and false negatives.
- The feature importance plot helps understand which features contribute most to fraud detection.

## Notes
- The dataset used in this project is highly imbalanced, so additional techniques like oversampling, undersampling, or SMOTE may be explored to improve performance.
- Other models like Gradient Boosting Classifier can be tested for comparison.

## License
This project is intended for educational purposes only.

