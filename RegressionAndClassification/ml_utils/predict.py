# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This module contains functions to predict outputs for linear and logistic regression models.

import numpy as np
from .sigmoid import sigmoid

def predict_linear(x, w, b):
    """
    Predicts output using linear regression model.
    Args:
        x: Input features, shape (m, n)
        w: Weights, shape (n,)
        b: Bias term (scalar)
    Returns:
        y_pred: Predicted outputs, shape (m,)
    """
    return np.dot(x, w) + b

def predict_logistic(x, w, b, threshold=0.5):
    """
    Predicts binary labels using logistic regression model.
    Args:
        x: Input features, shape (m, n)
        w: Weights, shape (n,)
        b: Bias term (scalar)
        threshold: Classification threshold (default 0.5)
    Returns:
        y_pred: Predicted labels (0 or 1), shape (m,)
    """
    probs = sigmoid(np.dot(x, w) + b)
    return (probs >= threshold).astype(int)
