# Machine-Learning-Course
Machine learning supported studies and applications
# Linear Regression Model with Gradient Descent

This program implements a simple linear regression model using gradient descent. It reads a dataset from a CSV file, normalizes the features, and iteratively updates the weights and bias to minimize the cost function. The program also visualizes the cost function over iterations to monitor the training progress.

## Dataset

The dataset is assumed to be in a CSV file named `dataset2.csv`, containing three columns:
- `X1`: The first feature.
- `X2`: The second feature.
- `Y`: The target variable.

Make sure the dataset file is located at the specified path (`C:/Users/New-Phone/Desktop/repos/Machine-Learning-Course/code/dataset2.csv`). Adjust the path if needed.

## Program Overview

1. **Load the Dataset**: The program reads the CSV file using `pandas` and extracts the features (`X1`, `X2`) and the target variable (`Y`).
2. **Feature Normalization**: The features are normalized to have a mean of 0 and a standard deviation of 1, improving the performance of gradient descent.
3. **Parameter Initialization**: Random values are initialized for the weights (`w1`, `w2`) and the bias (`b`).
4. **Gradient Descent**: 
    - The program iterates over a set number of times (2000 by default) to minimize the cost function using gradient descent. 
    - The cost function is the Mean Squared Error (MSE) between the predicted and actual values.
    - The weights and bias are updated using the gradients calculated from the MSE.
5. **Loss Visualization**: The program plots the cost function over iterations, allowing the user to visualize the model's convergence.

## Dependencies

The program requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib
