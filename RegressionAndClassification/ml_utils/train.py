# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This module contains functions to train linear and logistic regression models using gradient descent.
# It includes regularization to prevent overfitting.
import numpy as np
from .cost_linear import compute_cost_linear_regression
from .cost_logistic import compute_cost_logistic_regression
from .gradient_linear import compute_gradient_linear_regression
from .gradient_logistic import compute_gradient_logistic_regression
# ------------------------------------------------------------------------------
#  Training Functions for Linear and Logistic Regression
# ------------------------------------------------------------------------------
# These functions implement gradient descent to optimize the weights and bias
# for both linear and logistic regression models.
# They include regularization to help prevent overfitting.
# ------------------------------------------------------------------------------
def train_linear_regression(x, y, w_init, b_init, alpha, num_iters, lambda_=1.0):
    w = w_init.copy()
    b = b_init
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_linear_regression(x, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0 or i == num_iters - 1:
            cost = compute_cost_linear_regression(x, y, w, b, lambda_)
            cost_history.append(cost)
            print(f"Iter {i}: Cost={cost:.4f}")

    return w, b, cost_history

def train_logistic_regression(x, y, w_init, b_init, alpha, num_iters, lambda_=1.0):
    w = w_init.copy()
    b = b_init
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic_regression(x, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0 or i == num_iters - 1:
            cost = compute_cost_logistic_regression(x, y, w, b, lambda_)
            cost_history.append(cost)
            print(f"Iter {i}: Cost={cost:.4f}")

    return w, b, cost_history
