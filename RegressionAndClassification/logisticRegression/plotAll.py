# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def plot_all(X, y, w, b, J_history):
    """
    Combines 2 plots side-by-side:
    1. Cost function history during gradient descent
    2. Feature scatter with class labels and decision boundary
    
    Args:
        X: Training features, shape (m, 2)
        y: Labels (0 or 1), shape (m,)
        w: Learned weights, shape (2,)
        b: Learned bias (scalar)
        J_history: List of cost values during training
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # -----------------------------
    # Plot 1: Cost function history
    # -----------------------------
    axs[0].plot(range(len(J_history)), J_history, label='Cost Function J(w, b)', color='tab:blue')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Cost J(w, b)')
    axs[0].set_title('Cost Function History')
    axs[0].grid(True)
    axs[0].legend()

    # -----------------------------
    # Plot 2: Input features + decision boundary
    # -----------------------------
    scatter = axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].set_title('Input Features and Output Labels')

    # Plot decision boundary: w0 * x0 + w1 * x1 + b = 0 → x1 = -(b + w0 * x0) / w1
    x_vals = np.array([np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5])
    y_vals = -(b + w[0] * x_vals) / w[1]
    axs[1].plot(x_vals, y_vals, 'k--', label='Decision Boundary')

    # Legend for classes and boundary
    legend1 = axs[1].legend(*scatter.legend_elements(), title="Classes")
    axs[1].add_artist(legend1)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
