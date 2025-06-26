# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This module contains a function to compute the gradients for regularized logistic regression.
import numpy as np
from .sigmoid import sigmoid  # Importing sigmoid function from sigmoid.py
#Gradient function for regularized logistic regression
def compute_gradient_logistic_regression(x, y, w, b, lambda_=1.0):
    """
    Computes the gradients of the regularized cost function for logistic regression.
    Args:
        x: Training features, shape (m, n)
        y: Labels, shape (m,)
        w: Weights, shape (n,)
        b: Bias term (scalar)
        lambda_: Regularization parameter (default is 1.0)
    Returns:
        dj_dw: Gradient with respect to weights, shape (n,)
        dj_db: Gradient with respect to bias (scalar)
    """
    m, n = x.shape                             # m = number of examples, n = number of features
    dj_dw = np.zeros(n, dtype=np.longdouble)   # ∂J/∂w initialized to zeros
    dj_db = 0.0                                 # ∂J/∂b initialized to
    for i in range(m):                          # Loop over training examples
        z_i = np.dot(x[i], w) + b               # z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b
        f_wb_i = sigmoid(z_i)           # f_wb⁽ⁱ⁾ = sigmoid(z⁽ⁱ⁾) → predicted probability
        error = f_wb_i - y[i]                   # (f_wb - y) part of the gradient formula
        dj_dw += error * x[i]                   # Accumulate ∂J/∂w: add (f_wb - y) * x⁽ⁱ⁾
        dj_db += error                          # Accumulate ∂J/∂b: add (f_wb - y)
    dj_dw /= m                                  # Final average: (1/m) * Σ[∂J/∂w]   
    dj_db /= m                                  # Final average: (1/m) * Σ[∂J/∂b]
    # Add regularization term: (λ / m) * w
    dj_dw += (lambda_ / m) * w                 # ∂J/∂w with regularization
    return dj_dw, dj_db                         # Return the gradients