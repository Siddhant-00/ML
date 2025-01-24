# Decision Tree Pre-pruning with GridSearchCV

This project demonstrates how to implement and evaluate a decision tree classifier with pre-pruning techniques on the Iris dataset. The primary objective is to optimize the model's hyperparameters using GridSearchCV to achieve better accuracy and generalization.

## Requirements

To run this project, you will need the following Python libraries:

- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Workflow

### 1. Importing Required Libraries

We import essential libraries such as pandas, matplotlib, seaborn, and scikit-learn to handle data manipulation, visualization, and modeling.

### 2. Loading the Iris Dataset

The Iris dataset is loaded from scikit-learn’s built-in datasets module and also using seaborn for additional exploration.

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

### 3. Data Preparation

We separate the independent features (X) and the target variable (y). The dataset is then split into training and testing sets using `train_test_split`:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### 4. Training a Decision Tree Classifier

We initialize a decision tree classifier with a predefined maximum depth to prevent overfitting:

```python
from sklearn.tree import DecisionTreeClassifier
treemodel = DecisionTreeClassifier(max_depth=2)
treemodel.fit(X_train, y_train)
```

### 5. Visualizing the Decision Tree

We visualize the decision tree structure using Matplotlib:

```python
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel, filled=True)
```

### 6. Making Predictions

The model is used to predict on the test set, and the accuracy score and classification report are computed:

```python
from sklearn.metrics import accuracy_score, classification_report
y_pred = treemodel.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)
print(classification_report(y_pred, y_test))
```

### 7. Hyperparameter Tuning with GridSearchCV

To optimize the decision tree classifier, we use GridSearchCV to search over multiple hyperparameter combinations:

```python
parameter = {
 'criterion': ['gini', 'entropy', 'log_loss'],
 'splitter': ['best', 'random'],
 'max_depth': [1, 2, 3, 4, 5],
 'max_features': ['auto', 'sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV
treemodel = DecisionTreeClassifier()
cv = GridSearchCV(treemodel, param_grid=parameter, cv=5, scoring='accuracy')
cv.fit(X_train, y_train)
```

### 8. Evaluating the Best Model

The best hyperparameters and the corresponding performance metrics are retrieved:

```python
print(cv.best_params_)
y_pred = cv.predict(X_test)
score = accuracy_score(y_pred, y_test)
print(score)
print(classification_report(y_pred, y_test))
```

## Results

The model’s accuracy and classification report provide insights into its performance. By using GridSearchCV, the best combination of hyperparameters is selected to improve the decision tree’s predictive capabilities.

## Conclusion

This project demonstrates the implementation of pre-pruning in decision tree classification and highlights the benefits of hyperparameter tuning with GridSearchCV. By controlling the tree’s depth and other parameters, we can prevent overfitting and enhance the model’s generalization ability.

