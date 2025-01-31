# k-Nearest Neighbors (k-NN) Classifier for Diabetes Prediction

## Overview
This project implements a k-Nearest Neighbors (k-NN) classifier to predict the outcome of diabetes diagnosis based on patient data. The model is trained and evaluated using a dataset containing various health-related features.

## Dataset
The dataset is loaded from an Excel file: `diabetes.csv.xlsx`. It includes features such as:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Target variable: 0 = No diabetes, 1 = Diabetes)

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

## Steps in the Implementation

1. **Load the dataset**: Read the Excel file into a Pandas DataFrame.
2. **Data Preprocessing**:
   - Extract feature variables (`X`) and target variable (`y`).
   - Split the dataset into training and testing sets (60% training, 40% testing).
3. **Model Training & Evaluation**:
   - Train a k-NN classifier with different values of `k`.
   - Evaluate accuracy on training and testing sets.
   - Plot accuracy vs. number of neighbors.
4. **Performance Metrics**:
   - Compute and plot the confusion matrix.
   - Generate the ROC curve and compute the Area Under Curve (AUC).
5. **Hyperparameter Tuning**:
   - Use `GridSearchCV` to find the optimal `n_neighbors` value.

## Usage
Run the script in a Python environment (Jupyter Notebook, Google Colab, or a local script) to:
- Train and evaluate the k-NN model.
- Determine the optimal number of neighbors for better performance.
- Visualize results through accuracy and ROC curve plots.

## Results
- The model's ROC AUC score was approximately **0.7345**.
- The optimal number of neighbors was determined using **GridSearchCV**.

## License
This project is open-source and can be modified for learning and research purposes.

## Author
[Siddhant Jain]

