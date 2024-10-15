import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data = pd.read_csv('C:/Users/New-Phone/Desktop/repos/Machine-Learning-Course/code/dataset2.csv')

# Extract features and output from the dataset
X1 = data['X1'].values
X2 = data['X2'].values
Y = data['Y'].values

# Stack X1 and X2 into a single matrix X
X = np.column_stack((X1, X2))

# Normalize features (feature scaling)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Initialize parameters (weights and bias)
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Hyperparameters
learning_rate = 0.01
iterations = 2000

# To store the loss values for each iteration
Costs = []

# Gradient Descent Algorithm
for iteration in range(iterations):
    # Compute predictions
    Y_pred = w1 * X[:, 0] + w2 * X[:, 1] + b
    
    # Calculate loss (Mean Squared Error)
    cost = np.mean((Y - Y_pred) ** 2)
    Costs.append(cost)
    
    # Calculate gradients
    d_w1 = -2 * np.mean((Y - Y_pred) * X[:, 0])
    d_w2 = -2 * np.mean((Y - Y_pred) * X[:, 1])
    d_b = -2 * np.mean(Y - Y_pred)
    
    # Update weights and bias
    w1 -= learning_rate * d_w1
    w2 -= learning_rate * d_w2
    b -= learning_rate * d_b
    
    # Print the loss and parameters every 100 iteration
    if iteration % 100 == 0:
        print(f'iteration {iteration}: cost = {cost:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}')

# Final parameters
print(f'Final weights: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}')

# Plotting the loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), Costs, label='Cost')
plt.xlabel('Iterations')
plt.ylabel('Costs')
plt.title('Cost Function over Iterations')
plt.legend()
plt.show()
