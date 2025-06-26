# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Cost Function for Linear Regression
#
# The cost function J(w, b) is used to evaluate how well the linear model
# predicts the actual target values (y).
#
# Equation:
#     J(w, b) = (1 / 2m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i])²
#
# where:
#     f_wb(x[i]) = w * x[i] + b    # the prediction of our model
#     y[i]         = actual target value
#     m            = number of examples
#
# Notes:
# - (f_wb(x[i]) - y[i])² is the squared error between prediction and target.
# - The sum of squared errors is averaged and scaled by 1/(2m) to compute cost.
# - In math notation, index i starts from 1 to m, but in code we use 0 to m-1.
# ------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3, 5, 8, 10])  # Input variable (size in 1000 square feet)
y_train = np.array([300.0, 500.0, 600, 700, 800, 1000])  # Target (price in 1000s of dollars)

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray (m,)): Input data, m examples
        y (ndarray (m,)): Target values
        w (scalar): Weight parameter
        b (scalar): Bias parameter
    
    Returns:
        float: The cost of using w, b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]  # Number of training examples
    if m == 0:
        raise ValueError("Input array x is empty.")
    if not isinstance(w, (int, float)):
        raise TypeError("Weight w must be a number.")
    if not isinstance(b, (int, float)):
        raise TypeError("Bias b must be a number.")
    if x.ndim != 1:
        raise ValueError("Input x must be a 1-dimensional array.")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input x or y contains NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input x or y contains infinite values.")

    #  J(w, b) = (1 / 2m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i])² where f_wb(x[i] = w * x[i] + b
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    # Average cost scaled by 1/(2m)
    total_cost = (1 / (2 * m)) * cost_sum  # Average cost scaled by 1/(2m)

    if np.isnan(total_cost) or np.isinf(total_cost):
        raise ValueError("Computed cost is NaN or infinite, check input values.")
   
    return total_cost

# Example usage: using same w and b as in the model representation
w = 80.0  # Example weight
b = 250.0  # Example bias

cost = compute_cost(x_train, y_train, w, b)
print(f"Computed cost: {cost}")