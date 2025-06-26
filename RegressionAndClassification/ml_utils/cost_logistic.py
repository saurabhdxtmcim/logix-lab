# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This module contains a function to compute the regularized cost for logistic regression.
import numpy as np
from .sigmoid import sigmoid  # Importing sigmoid function from sigmoid.py
# ------------------------------------------------------------------------------
# Regularized Logistic Regression Cost Function
# ------------------------------------------------------------------------------
# Regularization helps prevent overfitting by discouraging large weight values.
# It does this by adding a penalty term to the cost function.
# The total cost function is made up of two parts:
# 1. Logistic loss (same as before):
#     J(w, b) = (1 / m) * Σ [ -y * log(f(x)) - (1 - y) * log(1 - f(x)) ]
#
#     Where:
#       - m = number of training examples
#       - f(x) = sigmoid(w · x + b) → predicted probability

# 2️. Regularization term (penalty on large weights):
#     (λ / 2m) * Σ (wⱼ²)
#
#     Where:
#       - λ is the regularization strength (lambda)
#       - This term **does NOT include the bias b**

# So the **final cost function** becomes:
#     J(w, b) = [logistic loss] + [regularization penalty]

#  Why do this?
# - Helps avoid overfitting by keeping weights small
# - Encourages simpler models that generalize better

def compute_cost_logistic_regression(x, y, w, b, lambda_=1.0):
    """    
    Computes the regularized cost for logistic regression.
    Args:
        x: Training features, shape (m, n)
        y: Labels, shape (m,)
        w: Weights, shape (n,)
        b: Bias term (scalar)
        lambda_: Regularization parameter (default is 1.0)
    Returns:
        Regularized cost J(w, b)
    """
    m = x.shape[0]                             # m = number of training examples
    cost = 0.0                                 # Initialize cost accumulator
    for i in range(m):
        z_i = np.dot(x[i], w) + b              # z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b → model prediction
        f_wb_i = sigmoid(z_i)           # f_wb⁽ⁱ⁾ = sigmoid(z⁽ⁱ⁾) → predicted probability
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i) # Add logistic loss: -y * log(f_wb) - (1 - y) * log(1 - f_wb)
    # Average cost over all examples: (1 / m) * Σ logistic loss
    cost /= m                                  # (1 / m) * Σ logistic loss → unregularized cost
    # Regularization term: (λ / 2m) * Σ(wⱼ²)
    reg_term = (lambda_ / (2 * m)) * np.sum(w ** 2)  # Excludes bias term b
    return cost + reg_term                     # Total cost = unregularized cost + regularization