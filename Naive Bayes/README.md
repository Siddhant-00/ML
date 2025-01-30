# Spam Classification using Naive Bayes

## Overview
This project implements a spam classifier using a Naive Bayes model. The dataset used for training and testing is the `spam.csv` file, which contains labeled messages as either 'spam' or 'ham' (not spam). The classification is performed using the `sklearn` library, and data preprocessing involves removing punctuation and stopwords.

## Requirements
Ensure you have the following Python libraries installed before running the code:

- numpy
- pandas
- matplotlib
- seaborn
- nltk
- sklearn

You can install any missing dependencies using:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```

## Dataset
The dataset is loaded from a CSV file, and only two columns are used:
- `label`: Indicates whether a message is 'spam' or 'ham'.
- `message`: The actual text content of the message.

## Data Preprocessing
1. Convert labels into categorical variables.
2. Compute the length of each message for exploratory analysis.
3. Remove punctuation and stopwords using NLTK.
4. Convert text into a numerical feature matrix using `CountVectorizer`.

## Model Training
1. Split the dataset into training (80%) and testing (20%) subsets.
2. Convert text messages into a sparse matrix representation.
3. Train a Gaussian Naive Bayes model using the training set.

## Model Evaluation
1. Use the trained model to predict spam/ham classifications for both training and testing sets.
2. Evaluate the model using a confusion matrix and classification report.

## Running the Code
1. Ensure `spam.csv` is correctly placed in the specified directory.
2. Run the Python script or execute the cells in a Jupyter Notebook.
3. Check the output for model performance metrics.

## Output
The final output includes:
- Confusion matrices for both training and testing datasets.
- Classification reports displaying precision, recall, and F1-score.

## Notes
- The current model uses a simple Naive Bayes approach. Consider experimenting with different vectorization methods (e.g., TF-IDF) and models (e.g., SVM, deep learning) to improve accuracy.
- The dataset may need further cleaning to handle special characters and typos.

## License
This project is open-source and can be used for educational purposes.

