# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


def gradient_descent_sklearn(x_train, y_train, max_iter=1000):
    """
    Perform logistic regression using scikit-learn's LogisticRegression.
    Args:
        x_train: Training features, shape (m, n)
        y_train: True labels, shape (m,)
        learning_rate: Learning rate for gradient descent (not used in sklearn)
        max_iter: Maximum number of iterations for convergence
    Returns:
        model: Trained LogisticRegression model
        cost_history: List of cost values at each iteration
    """
    # Create a logistic regression model
    model = LogisticRegression(solver='lbfgs', max_iter=max_iter, random_state=42)
    
    # Fit the model to the training data
    model.fit(x_train, y_train)
    # Predict probabilities for the training data
    y_prid = model.predict(x_train)
    print("Prediction on training set:", y_prid)
    return model


# Example usage
if __name__ == "__main__":
    # Sample training data
    x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1, ])
    # Perform gradient descent using scikit-learn
    model = gradient_descent_sklearn(x_train, y_train)
    
    print("Learned weights:", model)
    print("Learned intercept (bias):", model.intercept_)
    # Calculate the log loss on the training set
    y_prob = model.predict_proba(x_train)[:, 1]  # Get probabilities for the positive class
    cost = log_loss(y_train, y_prob)
    print("Log loss on training set:", cost)
    # Plotting the decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
    plt.xlabel('Feature 1') 
    plt.ylabel('Feature 2')
    plt.title('Training Data with Decision Boundary')
    # Plot decision boundary
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.colorbar()
    plt.show()
    