# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This module contains a function to compute the gradients for regularized linear regression.
import numpy as np
# ------------------------------------------------------------------------------
#  Gradient Descent with Regularization (Linear/Logistic Regression)
# ------------------------------------------------------------------------------

# The update rule for gradient descent **does not change** with regularization.
# You still update weights and bias like this:
#
#     w_j = w_j - α * ∂J/∂w_j
#     b   = b   - α * ∂J/∂b
#
# The only difference is: how we compute the gradients (∂J/∂w_j and ∂J/∂b)

# ----------------------------------------------------------------------
# For both Linear and Logistic Regression, the gradients become:
#
# ∂J/∂w_j = (1 / m) * Σ (f_wb(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x_j⁽ⁱ⁾   +   (λ / m) * w_j
# ∂J/∂b   = (1 / m) * Σ (f_wb(x⁽ⁱ⁾) - y⁽ⁱ⁾)
#
#  The extra term (λ / m) * w_j is the **regularization penalty**
#     - It "pushes" weights toward zero to reduce overfitting
#     - Note: bias `b` is NOT regularized (only weights `w`)

# ----------------------------------------------------------------------
#  Model predictions f_wb(x⁽ⁱ⁾) differ by type of regression:
# - For Linear Regression:     f(x) = w · x + b
# - For Logistic Regression:   f(x) = sigmoid(w · x + b)
# Regularization makes gradient descent favor **simpler models** by discouraging large weights.
# --------------------------------------------------------------------------

# Gradient function for regularized linear regression
def compute_gradient_linear_regression(x, y, w, b, lambda_=1.0):
    """
    Computes the gradients of the regularized cost function for linear regression.
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
    dj_db = 0.0                                 # ∂J/∂b initialized to 0
    for i in range(m):                          # Loop over training examples
        z_i = np.dot(x[i], w) + b               # z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b
        f_wb_i = z_i                            # f_wb⁽ⁱ⁾ = z⁽ⁱ⁾ for linear regression
        error = f_wb_i - y[i]                   # (f_wb - y) part of the gradient formula
        dj_dw += error * x[i]                   # Accumulate ∂J/∂w: add (f_wb - y) * x⁽ⁱ⁾
        dj_db += error                          # Accumulate ∂J/∂b: add (f_wb - y)  
    dj_dw /= m                                  # Final average: (1/m) * Σ[∂J/∂w]
    dj_db /= m                                  # Final average: (1/m) * Σ[∂J/∂b]
    # Add regularization term: (λ / m) * w
    dj_dw += (lambda_ / m) * w                 # ∂J/∂w with regularization
    return dj_dw, dj_db                         # Return the gradients