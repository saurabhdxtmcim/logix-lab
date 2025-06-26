# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt 
from sigmoid import sigmoid

# ------------------------------------------------------------------------------
# Visualizing Binary Classification Data for Logistic Regression
# ------------------------------------------------------------------------------

# Sample input features (x₀, x₁) and binary labels y ∈ {0, 1}
x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0,0,0,1,1,1]).reshape(-1, 1)  # Reshape to a column vector

# Plot the data points with class-specific markers
fig, ax = plt.subplots()
pos = y.flatten() == 1
neg = y.flatten() == 0
ax.scatter(x[pos, 0], x[pos, 1], marker='x', c='red', s=100, label='y=1')   # Positive class
ax.scatter(x[neg, 0], x[neg, 1], marker='o', edgecolors='blue', facecolors='none', s=100, label='y=0')  # Negative class
ax.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Points')
plt.show()

# ------------------------------------------------------------------------------
# Logistic Regression Prediction Rule (Simplified Overview)
# ------------------------------------------------------------------------------
# Logistic regression computes:      z = w·x + b
# The sigmoid function maps z to a probability: g(z) = 1 / (1 + e^(-z))
# The output f(x) = g(z) is interpreted as the probability that y = 1

# Decision rule:
# - If f(x) ≥ 0.5, predict y = 1
# - If f(x) <  0.5, predict y = 0
# The decision boundary lies where f(x) = 0.5 ⇨ this means z = 0 ⇨ w·x + b = 0

# ------------------------------------------------------------------------------
# Plotting the Sigmoid Curve
# ------------------------------------------------------------------------------
# Generate input z values from -10 to 10 to visualize the S-shaped sigmoid
z = np.arange(-10, 11)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(z, sigmoid(z), c="b")  # Plot sigmoid curve

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')

# Highlight decision threshold at g(z) = 0.5 (which occurs at z = 0)
ax.axhline(0.5, color='red', linestyle='--', label='Decision Boundary (f(x) = 0.5)')
ax.axvline(0, color='green', linestyle='--', label='z = 0')
ax.legend()
plt.show()

# ------------------------------------------------------------------------------
# Plotting the Logistic Regression Decision Boundary
# ------------------------------------------------------------------------------
# Example trained model: f(x) = g(w₀·x₀ + w₁·x₁ + b)
# Using: w₀ = 1, w₁ = 1, b = -3
# So, z = x₀ + x₁ - 3 ⇒ decision boundary is when z = 0:
#        x₀ + x₁ - 3 = 0  ⇨ x₁ = 3 - x₀

x0 = np.arange(0, 6)           # Values for x₀
x1 = 3 - x0                    # Corresponding decision boundary line for x₁

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(x0, x1, c = "b")         # Plot the decision boundary line
ax.axis([0, 4, 0, 3.5])
ax.fill_between(x0, x1, alpha = 0.2)  # Shade region below the line

# Plot original classification data on the same graph
def plot_data(X, y, ax):
    pos = y.flatten() == 1
    neg = y.flatten() == 0
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', c='red', s = 100, label='y=1')
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', edgecolors='blue', facecolors='none', s = 100, label = 'y=0')
    ax.legend()

plot_data(x, y, ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.title('Decision Boundary (x₁ = 3 - x₀)')
plt.show()
