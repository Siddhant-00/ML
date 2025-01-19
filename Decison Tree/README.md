# Humidity Prediction Using Morning Sensor Signals

## Project Overview
This project aims to predict whether the humidity will be high at 3 PM using morning sensor signals. The dataset used in this project contains various weather-related features recorded at 9 AM. The target variable is `high_humidity_3pm`, which indicates whether the humidity is high (1) or not (0) at 3 PM.

## Dataset
- **File**: `daily_weather.csv`
- **Target Variable**: `high_humidity_3pm`
- **Features**:
  - `air_pressure_9am`
  - `air_temp_9am`
  - `avg_wind_direction_9am`
  - `avg_wind_speed_9am`
  - `max_wind_direction_9am`
  - `max_wind_speed_9am`
  - `rain_accumulation_9am`
  - `rain_duration_9am`
  - `relative_humidity_9am`

## Requirements
Ensure you have the following Python packages installed:
- pandas
- scikit-learn
- matplotlib
- graphviz

To install the required packages, run:
```bash
pip install pandas scikit-learn matplotlib graphviz
```

## Steps

### 1. Data Loading and Exploration
- Load the dataset using pandas.
- Explore the dataset using functions like `.head()`, `.columns`, `.shape`, and `.describe()`.
- Check the distribution of the target variable using `.value_counts()`.

### 2. Data Cleaning
- Remove rows with null values using `data.dropna(inplace=True)`.

### 3. Data Preparation
- Define the dependent variable (`high_humidity_3pm`) and independent variables (features recorded at 9 AM).
- Split the dataset into training and testing sets using `train_test_split` from scikit-learn with a test size of 33%.

### 4. Model Training
- Train a Decision Tree Classifier using the `DecisionTreeClassifier` class from scikit-learn with the following parameters:
  - `criterion='entropy'`
  - `max_leaf_nodes=10`
  - `random_state=0`

### 5. Model Evaluation
- Use the trained model to make predictions on the test set.
- Evaluate the model's accuracy using `accuracy_score` from scikit-learn.

### 6. Visualization
- Visualize the decision tree using the `plot_tree` function from scikit-learn.
- Display the tree with:
  - Feature names
  - Class names
  - Filled nodes for better visualization

## Example Output
- The accuracy of the classifier is calculated and printed as a percentage.
- A visual representation of the decision tree is displayed using matplotlib.

## Instructions to Run
1. Clone or download this repository.
2. Update the `data` variable to point to the path of the `daily_weather.csv` file.
3. Run the script in a Jupyter Notebook or a Python IDE that supports inline plotting.
4. Observe the accuracy and decision tree visualization.

## Key Functions
- `train_test_split`: Splits the dataset into training and testing sets.
- `DecisionTreeClassifier.fit`: Trains the decision tree on the training data.
- `accuracy_score`: Measures the accuracy of the classifier.
- `plot_tree`: Visualizes the decision tree structure.

## Results
- The decision tree classifier's performance is evaluated based on accuracy.
- The decision tree is visualized to understand the model's decision-making process.

## Dependencies
- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- graphviz

---



