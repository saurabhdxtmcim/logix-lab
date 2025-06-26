# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This script demonstrates the training and prediction of linear and logistic regression models.
# It uses simulated data for both regression tasks.
# ----------------------------------------------------------------------------------
import numpy as np
from ml_utils.train import train_linear_regression, train_logistic_regression
from ml_utils.predict import predict_linear, predict_logistic

# Simulated data
np.random.seed(42)
X = np.random.rand(100, 3)
y_logistic = (np.sum(X, axis=1) > 1.5).astype(int)
y_linear = np.sum(X, axis=1) + np.random.randn(100) * 0.1

# Hyperparameters
alpha = 0.1
num_iters = 1000
lambda_ = 0.7
w_init = np.zeros(X.shape[1])
b_init = 0.0

# Train Linear Regression
print("\nTraining Linear Regression:")
w_lin, b_lin, _ = train_linear_regression(X, y_linear, w_init, b_init, alpha, num_iters, lambda_)
y_pred_lin = predict_linear(X, w_lin, b_lin)
print("First 5 predictions (linear):", y_pred_lin[:5])

# Train Logistic Regression
print("\nTraining Logistic Regression:")
w_log, b_log, _ = train_logistic_regression(X, y_logistic, w_init, b_init, alpha, num_iters, lambda_)
y_pred_log = predict_logistic(X, w_log, b_log)
print("Logistic Accuracy:", np.mean(y_pred_log == y_logistic))

import matplotlib.pyplot as plt

# Visualize predictions and regressions
plt.figure(figsize=(10, 5))

# For linear regression: plot true vs predicted 
plt.subplot(1, 2, 1)
plt.scatter(y_linear, y_pred_lin, alpha=0.7)
plt.plot([y_linear.min(), y_linear.max()], [y_linear.min(), y_linear.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression\nTrue vs Predicted')

# For logistic regression: plot predicted probabilities vs true labels
plt.subplot(1, 2, 2)
plt.scatter(np.sum(X, axis=1), y_logistic, label='True Label', alpha=0.5)
plt.scatter(np.sum(X, axis=1), y_pred_log, label='Predicted', marker='x', alpha=0.5)
plt.xlabel('Sum of Features')
plt.ylabel('Class')
plt.title('Logistic Regression\nTrue vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()