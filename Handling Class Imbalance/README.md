# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using various machine learning techniques, including traditional classifiers and deep learning models. The dataset contains a highly imbalanced distribution of fraudulent and non-fraudulent transactions, which is handled through different resampling techniques.

## Dataset
The dataset used is `creditcard.csv`, which consists of transaction details with the target variable `Target` (0 for non-fraudulent transactions and 1 for fraudulent transactions).

## Dependencies
Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow
```

## Data Preprocessing
1. Load the dataset using Pandas.
2. Analyze class distribution and visualize the target variable.
3. Handle class imbalance using:
   - Random Under-Sampling (RUS)
   - Random Over-Sampling (ROS)
   - Synthetic Minority Over-sampling Technique (SMOTE)

## Machine Learning Models
Several classification models are implemented:

1. **K-Nearest Neighbors (KNN)**
   - Uses Minkowski distance with `k=5`.

2. **Support Vector Machine (SVM)**
   - Uses class weight balancing to handle the imbalance.

3. **XGBoost Classifier**
   - A gradient boosting algorithm for better performance.

4. **Deep Neural Network (DNN)**
   - A sequential model built using TensorFlow/Keras.
   - Multiple dense layers with dropout for regularization.

## Model Evaluation
Models are evaluated using:
- **Accuracy Score**
- **ROC-AUC Score**
- **Confusion Matrix**

## Training and Validation
- The dataset is split into training and testing sets (80-20 split).
- Resampling methods are applied to improve model performance.
- Neural networks are trained with `batch_size=512` for 20 epochs.

## Results
- The evaluation metrics are printed for all models.
- The training accuracy and validation accuracy of the deep learning model are plotted for analysis.

## Usage
1. Place `creditcard.csv` in the specified directory.
2. Run the script to preprocess data, train models, and evaluate results.
3. Compare different classifiers and select the best-performing model.

## Future Improvements
- Implement feature engineering to improve model performance.
- Explore ensemble learning techniques.
- Tune hyperparameters for better accuracy and ROC-AUC scores.

