# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import copy, math
import numpy as np
import matplotlib.pyplot as plt
from logisticLoss import compute_cost_logistic
from sigmoid import sigmoid
from plotAll import plot_all
# ------------------------------------------------------------------------------'

x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data for Logistic Regression')
plt.legend(['y=0', 'y=1'])
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------
#  Gradient Descent for Logistic Regression
# ------------------------------------------------------------------------------

# Goal: Minimize the cost function J(w, b) by updating weights w and bias b.

# Repeat until convergence:
#     w_j = w_j - α * ∂J(w, b)/∂w_j      for j = 0 to n-1
#     b   = b   - α * ∂J(w, b)/∂b

# Where α is the learning rate.

# ----------------------------------------------------------------------
# 🔍 How to Calculate the Gradients:
# ----------------------------------------------------------------------
# Step 1️⃣: Initialize weights and bias
# - Start with some initial guess values (e.g., zeros or small random numbers)
# Example:
#     w = [0, 0]
#     b = 0

# Step 2️⃣: Loop over all training examples
# For each training example (xᶦ, yᶦ):
#   1. Compute the prediction using the sigmoid function:
#        z     = w · xᶦ + b
#        f_wb  = 1 / (1 + exp(-z))
#
#   2. Measure the prediction error:
#        error = f_wb - yᶦ
#
#   3. For each weight w[j], update the gradient:
#        dj_dw[j] += error * xᶦ[j]
#
#   4. For the bias, accumulate the error:
#        dj_db += error

# Step 3️⃣: Average the gradients
# After looping through all m training examples:
#     dj_dw = dj_dw / m
#     dj_db = dj_db / m

# Step 4️⃣: Update weights and bias
# Use the gradients to move a small step in the opposite direction of the slope:
#     w = w - alpha * dj_dw
#     b = b - alpha * dj_db
#
# Here, alpha is the learning rate — controls how big each update step is.

# 🧪 Repeat this process for many iterations
# Keep updating w and b until they stop changing much.
# That means your model has *learned* the best fit for the training data!

def compute_gradient_logistic(x, y, w, b):
    """
    Computes the gradients of the logistic cost function with respect to weights and bias.
    Args:
        x: Training features, shape (m, n)  
        y: True labels, shape (m,)
        w: Weights, shape (n,)
        b: Bias term (scalar)
    Returns:
        dj_dw: Gradient with respect to weights, shape (n,)
        dj_db: Gradient with respect to bias (scalar)
    """
    m, n = x.shape                             # m = number of examples, n = number of features
    dj_dw = np.zeros(n, dtype=np.longdouble)   # ∂J/∂w initialized to zeros
    dj_db = 0.0                                 # ∂J/∂b initialized to 0

    for i in range(m):                          # Loop over training examples
        z_i = np.dot(x[i], w) + b               # z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b
        f_wb_i = sigmoid(z_i)                   # f_wb⁽ⁱ⁾ = sigmoid(z⁽ⁱ⁾)
        error = f_wb_i - y[i]                   # (f_wb - y) part of the gradient formula

        dj_dw += error * x[i]                   # Accumulate ∂J/∂w: add (f_wb - y) * x⁽ⁱ⁾
        dj_db += error                          # Accumulate ∂J/∂b: add (f_wb - y)

    dj_dw /= m                                  # Final average: (1/m) * Σ[∂J/∂w]
    dj_db /= m                                  # Final average: (1/m) * Σ[∂J/∂b]

    return dj_dw, dj_db                         # Return the gradients
# ------------------------------------------------------------------------------

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.

# Compute the gradients for the given parameters
dj_dw, dj_db = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print("Gradient with respect to weights (dj_dw):", dj_dw)
print("Gradient with respect to bias (dj_db):", dj_db)

# ------------------------------------------------------------------------------

def gradient_descent_logistic(x, y, w_init, b_init, alpha, num_iters):
    """
    Performs gradient descent to learn weights and bias for logistic regression.

    Args:
        x: Training features, shape (m, n)
        y: True labels, shape (m,)
        w_init: Initial weights, shape (n,)
        b_init: Initial bias (scalar)
        alpha: Learning rate (scalar)
        num_iters: Number of iterations for gradient descent (scalar)

    Returns:
        w: Learned weights, shape (n,)
        b: Learned bias (scalar)
    """
    w = copy.deepcopy(w_init)     # Initialize weights (make a copy to avoid changing the original)
    b = b_init                    # Initialize bias
    J_history = []               # To store the cost value at intervals (for visualization/debugging)

    # Loop for the specified number of iterations
    for i in range(num_iters):

        # Step 1: Compute the gradients of cost function J(w, b)
        # Using logistic regression gradient formulas
        dj_dw, dj_db = compute_gradient_logistic(x, y, w, b)

        # Step 2: Update weights and bias using the gradients
        # w = w - α * ∂J/∂w
        # b = b - α * ∂J/∂b
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Step 3 (optional): Monitor the cost every 100 iterations
        if i % 100 == 0:
            cost = compute_cost_logistic(x, y, w, b)  # Compute current cost J(w, b)
            J_history.append(cost)                    # Store it for later plotting/debugging
            print(f"Iteration {i}, Cost: {cost:.4f}, Weights: {w}, Bias: {b}")

    # Return the final learned parameters after all iterations
    return w, b, J_history
# ------------------------------------------------------------------------------

# Example usage of gradient descent for logistic regression
w_init = np.zeros(x_train.shape[1], dtype=np.longdouble)  # Initialize weights to zeros
b_init = 0.0  # Initialize bias to zero
alpha = 0.01
num_iters = 10000  # Number of iterations for gradient descent
w_learned, b_learned, J_history = gradient_descent_logistic(x_train, y_train, w_init, b_init, alpha, num_iters)
print("Learned weights:", w_learned)
print("Learned bias:", b_learned)
# ------------------------------------------------------------------------------

plot_all(x_train, y_train, w_learned, b_learned, J_history)