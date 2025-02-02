# Support Vector Machines (SVM) Implementation

## Overview
This project demonstrates the implementation of Support Vector Machines (SVM) using Python and Scikit-Learn. The code includes:
- Creating synthetic datasets
- Visualizing decision boundaries
- Understanding margin and support vectors
- Experimenting with different kernels (linear, radial basis function)

## Dependencies
Ensure you have the following Python libraries installed:
- `numpy`
- `matplotlib`
- `scipy`
- `seaborn`
- `scikit-learn`
- `ipywidgets`
- `mpl_toolkits`

You can install missing dependencies using:
```bash
pip install numpy matplotlib scipy seaborn scikit-learn ipywidgets
```

## Implementation Details

### 1. Data Generation and Visualization
- Used `make_blobs` to create synthetic datasets.
- Plotted the data points with different colors based on their class labels.

### 2. Decision Boundaries and Margins
- Plotted different linear decision boundaries.
- Highlighted the margins around the separating hyperplane.

### 3. Training SVM Models
- Used `SVC` from `sklearn.svm` with a linear kernel.
- Plotted the decision boundary and identified support vectors.

### 4. Exploring Kernel Tricks
- Implemented a nonlinear dataset using `make_circles`.
- Visualized feature transformation in 3D.
- Applied an RBF (Radial Basis Function) kernel to classify nonlinear data.

### 5. Effect of Regularization Parameter `C`
- Trained models with different values of `C`.
- Compared decision boundaries and the number of support vectors.

## Running the Code
Execute the Jupyter Notebook cell by cell or run the Python script to visualize SVM decision boundaries.

## References
- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Support Vector Machine Explanation](https://en.wikipedia.org/wiki/Support_vector_machine)

