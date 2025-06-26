# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Problem Statement: Multivariate Linear Regression for Housing Prices
#
# This dataset includes three training examples, each with four features:
# - Size of the house (in sqft)
# - Number of bedrooms
# - Number of floors
# - Age of the home (in years)
#
# Target variable:
# - Price (in 1000s of dollars)
#
# Training Data:
# | Size (sqft) | Bedrooms | Floors | Age | Price (1000s $) |
# |-------------|----------|--------|-----|-----------------|
# |    2104     |    5     |   1    |  45 |       460       |
# |    1416     |    3     |   2    |  40 |       232       |
# |     852     |    2     |   1    |  35 |       178       |
#
# Objective:
# - Build a linear regression model using these features to predict housing prices.
# - Example use case: Predict price for a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.
# ------------------------------------------------------------------------------

import copy, math
import numpy as np
import matplotlib.pyplot as plt
#from costFunction import compute_cost : dont need this import here, as we are defining it in this file and this one is not for 2D
np.set_printoptions(precision = 2)  # reduced display precision on numpy arrays

x_train = np.array([[2104, 5, 1, 45],
                   [1416, 3, 2, 40],
                   [852, 2, 1, 35]])  # Input features: Size, Bedrooms, Floors, Age 
y_train = np.array([460, 232, 178])  # Target variable: Price in 1000s of dollars

# data is stored in numpy array/matrix
print(f"X Shape: {x_train.shape}, X Type:{type(x_train)})")
print(x_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# any initial value for w and b: 𝐰 is a 1-D NumPy vector and b is scalar here
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# ------------------------------------------------------------------------------
# Model Prediction with Multiple Variables
#
# The linear model for multiple input features is defined as:
#     f_w,b(x) = w_0*x_0 + w_1*x_1 + ... + w_(n-1)*x_(n-1) + b
#
# In vector notation:
#     f_w,b(x) = w · x + b
# where:
#     - w and x are vectors of the same length (weights and features)
#     - "·" denotes the dot product between w and x
#
# This model generalizes the simple linear regression to multiple features.
# To demonstrate this, we will implement prediction using both the element-wise
# form and the vectorized (dot product) form.
#
# ------------------------------------------------------------------------------

# Single Prediction Element by Element
#
# Previously, prediction involved multiplying a single feature by its weight
# and adding a bias. To extend this to multiple features, we:
#   - Multiply each feature by its corresponding parameter (weight)
#   - Sum all the results
#   - Add the bias term at the end
#
# This can be implemented using a loop or vectorized using NumPy dot product.
# ------------------------------------------------------------------------------

def predict_single_loop(x, w, b):
    """
    Predicts the output for a single example using a loop.
    
    Args:
        x (ndarray (n,)): Input features for a single example
        w (ndarray (n,)): Weights for each feature
        b (scalar): Bias term
    
    Returns:
        float: Predicted value
    """
    
    prediction = 0.0
    for i in range(x.shape[0]):
        prediction += w[i] * x[i]
    return prediction + b 
  
# get a row from x_train to predict
x_example = x_train[0]  # Example input features for the first training example
# predict using the single loop method
predicted_value_loop = predict_single_loop(x_example, w_init, b_init)
print(f"Predicted value using single loop: {predicted_value_loop}")
# ------------------------------------------------------------------------------

#  f_w,b(x) = w_0*x_0 + w_1*x_1 + ... + w_(n-1)*x_(n-1) + b can be computed
# using the dot product of w and x, plus the bias term.
def predict_vectorized(x, w, b):
    """
    Predicts the output for a single example using vectorized operations.
    
    Args:
        x (ndarray (n,)): Input features for a single example
        w (ndarray (n,)): Weights for each feature
        b (scalar): Bias term
    
    Returns:
        float: Predicted value
    """
    
    return np.dot(w, x) + b

# predict using the vectorized method
predicted_value_vectorized = predict_vectorized(x_example, w_init, b_init)
print(f"Predicted value using vectorized method: {predicted_value_vectorized}")
# ------------------------------------------------------------------------------

# Both methods should yield the same result
if math.isclose(predicted_value_loop, predicted_value_vectorized):
    print("Both prediction methods yield the same result.")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Compute Cost with Multiple Variables
#
# The cost function J(w, b) for linear regression with multiple input features is:
#
#     J(w, b) = (1 / 2m) * Σ[i=0 to m-1] (f_w,b(x^(i)) - y^(i))²
#
# where:
#     - m is the number of training examples
#     - f_w,b(x^(i)) = w · x^(i) + b is the prediction for the i-th example
#     - w and x^(i) are vectors representing weights and input features, respectively
#     - y^(i) is the actual target value
#
# In contrast to previous labs where x was a scalar (single feature),
# x^(i) and w are now vectors, supporting multiple input features.
#
# This cost function measures the average squared error across all training examples.
# ------------------------------------------------------------------------------

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression with multiple features.
    
    Args:
        x (ndarray (m, n)): Input data, m examples with n features
        y (ndarray (m,)): Target values
        w (ndarray (n,)): Weights for each feature
        b (scalar): Bias term
    
    Returns:
        float: The cost of using w, b as the parameters for linear regression
               to fit the data points in x and y
    """
    
    m = x.shape[0]  # Number of training examples
    cost  = 0.0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        cost += (f_wb - y[i]) ** 2
    total_cost = (1 / (2 * m)) * cost  # Average cost scaled by 1/(2m)
    
    if np.isnan(total_cost) or np.isinf(total_cost):
        raise ValueError("Computed cost is NaN or infinite, check input values.")
    return total_cost

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(x_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

# ------------------------------------------------------------------------------
# Gradient Descent with Multiple Variables
#
# To optimize a multivariate linear regression model, gradient descent is used
# to iteratively update the model parameters (weights and bias):
#
# Update rule (for each feature j from 0 to n-1):
#     w_j := w_j - α * ∂J(w, b)/∂w_j
#     b   := b - α * ∂J(w, b)/∂b
#
# where:
#     - α is the learning rate
#     - J(w, b) is the cost function
#     - n is the number of features
#
# The partial derivatives (gradients) used in the update rule are computed as:
#     ∂J(w, b)/∂w_j = (1/m) * Σ[i=0 to m-1] (f_w,b(x^(i)) - y^(i)) * x_j^(i)
#     ∂J(w, b)/∂b   = (1/m) * Σ[i=0 to m-1] (f_w,b(x^(i)) - y^(i))
#
# Notes:
# - m is the number of training examples
# - f_w,b(x^(i)) is the predicted value for the i-th example
# - y^(i) is the actual target value
# - x_j^(i) is the j-th feature of the i-th example
#
# All parameters (weights and bias) must be updated **simultaneously** in each iteration.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Implementation Notes for Computing Gradients (∂J/∂w_j and ∂J/∂b)
#
# One approach to computing gradients for gradient descent with multiple variables:
#
# 1. Outer Loop:
#    - Loop over all m training examples.
#    - For each example, compute the prediction error: (f_w,b(x^(i)) - y^(i))
#
# 2. Inner Computations:
#    - Accumulate the error to compute ∂J(w, b)/∂b (gradient with respect to bias).
#    - Then, in a second loop over all n features:
#        - For each feature j, compute and accumulate the term:
#              ∂J(w, b)/∂w_j = error * x_j^(i)
#
# This double loop structure helps calculate gradients manually for educational clarity,
# SD NOTE: though vectorized implementations are preferred for efficiency in practice: but this one is easier to follow at least for me.
# x is 2 D array shape gives features and examples, so x[i] is a 1-D array of features for the i-th example.
# ------------------------------------------------------------------------------

def compute_gradient(x, y, w, b):
    """
    Computes the gradients of the cost function with respect to w and b.
    Args:
        x (ndarray (m, n)): Input data, m examples with n features
        y (ndarray (m,)): Target values
        w (ndarray (n,)): Weights for each feature
        b (scalar): Bias term
    
    Returns:
        tuple: Gradients (dj_dw, dj_db)
    """
    m, n = x.shape  # Number of training examples (m) and features (n)
    if m == 0 or n == 0:
        raise ValueError("Input array x is empty or has no features.")
    if not isinstance(w, np.ndarray) or w.ndim != 1 or w.shape[0] != n:
        raise TypeError("Weight w must be a 1-D NumPy array with shape (n,).")
    if not isinstance(b, (int, float)):
        raise TypeError("Bias b must be a number.")
    if x.ndim != 2 or x.shape[0] != m or x.shape[1] != n:
        raise ValueError("Input x must be a 2-dimensional array with shape (m, n).")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Input x or y contains NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Input x or y contains infinite values.")
    if np.any(y < 0):
        raise ValueError("Target y contains negative values, which is not valid for price in 1000s of dollars.")
    if np.any(x < 0):
        raise ValueError("Input x contains negative values, which is not valid for size in sqft or other features.")
    
    dj_dw = np.zeros(n)  # Initialize gradient for w
    dj_db = 0.0  # Initialize gradient for b

    for i in range(m):
        f_wb = (np.dot(x[i], w) + b) - y[i]  # Prediction error for the i-th example: (f_w,b(x^(i)) - y^(i)), where f_w,b(x^(i)) = w · x^(i) + b
        for j in range(n):
            dj_dw[j] += f_wb * x[i, j]  # Accumulate gradient for w_j
        dj_db += f_wb  # Accumulate gradient for b

    # Average the gradients
    dj_dw /= m
    dj_db /= m
    if np.isnan(dj_dw).any() or np.isinf(dj_dw).any():
        raise ValueError("Computed gradient dj_dw contains NaN or infinite values.")
    if np.isnan(dj_db) or np.isinf(dj_db):
        raise ValueError("Computed gradient dj_db is NaN or infinite.")
    return dj_dw, dj_db

# Example usage
gradientVal = compute_gradient(x_train, y_train, w_init, b_init)
print(f"Gradient w: {gradientVal[0]}, Gradient b: {gradientVal[1]}")

# ------------------------------------------------------------------------------
# Gradient Descent Function for Multiple Variables
## The gradient descent algorithm for multivariate linear regression updates
# the parameters w and b iteratively using the computed gradients.
# The update rule is:   
#     w_j := w_j - α * ∂J(w, b)/∂w_j
#     b   := b - α * ∂J(w, b)/∂b
## where:
#     - α is the learning rate
#     - ∂J(w, b)/∂w_j and ∂J(w, b)/∂b are the gradients computed in the previous step
#
# The algorithm continues until convergence or for a fixed number of iterations.
# ------------------------------------------------------------------------------
def gradient_descent(x, y, w, b, alpha, num_iterations, cost_function=compute_cost, compute_gradient=compute_gradient):
    """
    Performs gradient descent to learn w and b.
    
    Args:
        x (ndarray (m, n)): Input data, m examples with n features
        y (ndarray (m,)): Target values
        w (ndarray (n,)): Initial weights for each feature
        b (scalar): Initial bias term
        alpha (float): Learning rate
        num_iterations (int): Number of iterations for gradient descent
        cost_function (function): Function to compute cost
        compute_gradient (function): Function to compute gradients
    Returns:
        tuple: Final parameters (w, b), cost history, and parameter history
    """
    j_history = []
    w = copy.deepcopy(w)  # Ensure w is a copy to avoid modifying the original
    b = float(b)  # Ensure b is a float for consistency
    p_history = []

    for i in range(num_iterations):
        # Compute gradients
        dj_dw, dj_db = compute_gradient(x, y, w, b)  # ∂J(w, b)/∂w and ∂J(w, b)/∂b
        print(f"Iteration {i}: dj_dw = {dj_dw}, dj_db = {dj_db}")  # Debugging output for gradients
        # Update parameters
        w -= alpha * dj_dw  # w = w - α * ∂J(w, b) / ∂w
        b -= alpha * dj_db  # b = b - α * ∂J(w, b) / ∂b
        print(f"Iteration {i}: Updated w = {w}, Updated b = {b}")  # Debugging output for updated parameters
        # Compute cost for the current parameters
        cost = cost_function(x, y, w, b)
        j_history.append(cost)
        p_history.append((copy.deepcopy(w), float(b)))  # Store a copy of parameters

        # Print cost every 100 iterations for monitoring
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")
    return w, b, j_history, p_history

# Example usage
alpha = 0.0000001  # Learning rate
num_iterations = 10000  # Number of iterations for gradient descent
w_final, b_final, j_history, p_history = gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iterations)
print(f"Final parameters: w = {w_final}, b = {b_final}")    

# draw the cost history
plt.plot(j_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.grid()
plt.show()
