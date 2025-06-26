# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Why Not Use Squared Error for Logistic Regression?
# ------------------------------------------------------------------------------

# In linear regression, we use squared error:
#     J(w, b) = (1 / 2m) ∑ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
#
# This works because the squared error cost is convex.
# However, with logistic regression (where f(x) = sigmoid(w · x + b)),
# squared error becomes non-convex, making optimization unreliable.

# ------------------------------------------------------------------------------
# Logistic Loss (Cross-Entropy) for Binary Classification
# ------------------------------------------------------------------------------

# Logistic regression outputs probabilities for binary classes (y ∈ {0, 1}).
# We use **log loss** (aka **cross-entropy**) to evaluate predictions.

# ➤ Loss vs. Cost:
# - "Loss" is the error for one training example.
# - "Cost" is the average loss across the whole dataset.

# Logistic Loss Formula:
# For a prediction f(x) = sigmoid(w · x + b), and true label y ∈ {0, 1}:
#
#   loss(f(x), y) = {
#       -log(f(x))         if y = 1
#       -log(1 - f(x))     if y = 0
#   }
#
# General form (used in code):
#   loss = -y * log(f(x)) - (1 - y) * log(1 - f(x))

#  Intuition:
# - Loss is low when prediction is close to the true label.
# - Loss is high when the model is confidently wrong.

#  Benefits of Log Loss:
# - Works with probabilities.
# - Results in a **convex** cost function with sigmoid output.
# - Enables reliable convergence using gradient descent.

# ------------------------------------------------------------------------------
# 🧪 Visualizing the Training Data
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sigmoid import sigmoid  

# Simple dataset (1 feature)
x_train_1d = np.array([0., 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train_1d = np.array([0,  0, 0, 1, 1, 1], dtype=np.longdouble)

plt.scatter(x_train_1d, y_train_1d, c=y_train_1d, cmap='bwr', edgecolors='k')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.title('Binary Classification Data (1D)')
plt.legend(['y=0', 'y=1'])
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------
#  Logistic Loss and Cost Function Implementation
# ------------------------------------------------------------------------------

# Define training data with 2 features
x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Logistic loss for one example:
# loss(f_wb(x⁽ⁱ⁾), y⁽ⁱ⁾) = -y⁽ⁱ⁾ * log(f_wb(x⁽ⁱ⁾)) - (1 - y⁽ⁱ⁾) * log(1 - f_wb(x⁽ⁱ⁾))
# where:
#   f_wb(x⁽ⁱ⁾) = 1 / (1 + exp(-z⁽ⁱ⁾)), z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b

def compute_cost_logistic(x, y, w, b):
    """
    Computes average logistic loss (cost) over the training examples.

    Args:
        x: Training features, shape (m, n)
        y: True labels, shape (m,)
        w: Weights, shape (n,)
        b: Bias term (scalar)

    Returns:
        Average logistic cost across all m examples
    """
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost /= m
    return cost

# ------------------------------------------------------------------------------
# Testing the Cost Function with Different Parameters
# ------------------------------------------------------------------------------

w1 = np.array([1, 1])
b1 = -3
w2 = np.array([1, 1])
b2 = -4

print("Cost for b = -3:", compute_cost_logistic(x_train, y_train, w1, b1))
print("Cost for b = -4:", compute_cost_logistic(x_train, y_train, w2, b2))
