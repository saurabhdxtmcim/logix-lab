# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
#  Cost Function with Regularization (Logistic or Linear Regression)
# ------------------------------------------------------------------------------

# Regularization helps prevent overfitting by discouraging large weight values.
# It adds a penalty to the cost function that grows when weights get large.

# General regularized cost function:
#     J(w, b) = (1 / 2m) * Σ( f(x⁽ⁱ⁾) - y⁽ⁱ⁾ )² + (λ / 2m) * Σ( wⱼ² )
#
# For **linear regression**:
#     f(x⁽ⁱ⁾) = w · x⁽ⁱ⁾ + b   ← raw value (no activation)

# For **logistic regression**:
#     f(x⁽ⁱ⁾) = 1 / (1 + exp( - (w · x⁽ⁱ⁾ + b) ))   ← sigmoid of z
#
# This gives a probability between 0 and 1 for binary classification.

# ----------------------------------------
#  Term Breakdown:
# - m = number of training examples
# - w · x⁽ⁱ⁾ + b is the model’s prediction (z)
# - f(x⁽ⁱ⁾) is the predicted value (linear or sigmoid depending on model)
# - λ = regularization parameter (lambda)
# - Σ( wⱼ² ) is the sum of squares of the weights (but NOT the bias)
#
#  Regularization term: (λ / 2m) * Σ( wⱼ² )
#     - Encourages smaller weights
#     - Helps reduce overfitting
#     - Common in Ridge (L2) regularization
#
#  The bias `b` is not regularized — only weights `w`
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from .sigmoid import sigmoid  # Importing sigmoid function from sigmoid.py

def compute_cost_linear_regression(x, y, w, b, lambda_ = 1.0):
    """
    Computes the regularized cost for linear regression.
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
        f_wb_i = z_i                           # f_wb⁽ⁱ⁾ = z⁽ⁱ⁾ for linear regression
        cost += (f_wb_i - y[i]) ** 2           # Add squared error: (f_wb⁽ⁱ⁾ - y⁽ⁱ⁾)²
    cost /= (2 * m)                            # (1 / 2m) * Σ squared errors → unregularized cost
    # Regularization term: (λ / 2m) * Σ(wⱼ²)
    reg_term = (lambda_ / (2 * m)) * np.sum(w ** 2)  # Excludes bias term b
    return cost + reg_term                     # Total cost = unregularized cost + regularization