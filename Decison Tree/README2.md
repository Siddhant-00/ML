# Implementation of Pre-Pruning

This document provides a detailed explanation of the implementation of pre-pruning in a Decision Tree model using the Iris dataset. Pre-pruning aims to stop the tree growth early by setting constraints, such as maximum depth or minimum samples per split, to avoid overfitting.

## Requirements

Ensure you have the following libraries installed before running the code:

- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip if they are not already installed:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Dataset

The Iris dataset is used for this implementation. It is loaded using scikit-learn and seaborn for visualization.

## Steps

### 1. Importing Libraries
The essential libraries are imported:
- pandas and seaborn for data handling and visualization.
- matplotlib for plotting.
- scikit-learn for model implementation and evaluation.

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import seaborn as sns
```

### 2. Loading the Dataset
The Iris dataset is loaded using both scikit-learn and seaborn:
```python
iris = load_iris()
df = sns.load_dataset('iris')
```

### 3. Preparing Features and Targets
Independent (features) and dependent (target) variables are defined:
```python
X = df.iloc[:, :-1]
y = iris.target
```

### 4. Splitting Data
The dataset is split into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### 5. Model Training (Without Pre-Pruning)
A Decision Tree Classifier is trained on the training set:
```python
treemodel = DecisionTreeClassifier()
treemodel.fit(X_train, y_train)
```

### 6. Visualizing the Tree
The trained tree is visualized:
```python
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel, filled=True)
```

### 7. Predictions and Evaluation
The model's performance is evaluated using accuracy and classification reports:
```python
y_pred = treemodel.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)
print(classification_report(y_pred, y_test))
```

## Notes on Pre-Pruning
To implement pre-pruning, constraints such as `max_depth`, `min_samples_split`, or `min_samples_leaf` can be added to the `DecisionTreeClassifier`:

```python
treemodel = DecisionTreeClassifier(max_depth=3, min_samples_split=4)
treemodel.fit(X_train, y_train)
```

These constraints can improve the model's generalization by preventing overfitting.

## Outputs
- Visual representation of the decision tree.
- Accuracy score.
- Classification report showing precision, recall, and F1-score.

## Conclusion
This implementation demonstrates how to train and evaluate a decision tree classifier. Pre-pruning techniques can be applied by adding constraints during model initialization to improve generalization.

