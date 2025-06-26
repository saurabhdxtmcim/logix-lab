# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# FEATURE SCALING: Z-Score Normalization
#
# Feature scaling ensures all input features contribute equally to the model's learning.
# Without scaling, features with large ranges (e.g., house size in sqft) can dominate
# the gradient updates, causing instability or slow convergence in algorithms like
# gradient descent.
#
# We apply **Z-score normalization**, which standardizes each feature as:
#     x_scaled = (x - mean) / std_dev
# This transforms each feature to have a **mean of 0** and a **standard deviation of 1**.
#
# For feature j:
#   μ_j = (1/m) * Σ x_j^(i)     -> mean of feature j
#   σ_j = sqrt((1/m) * Σ (x_j^(i) - μ_j)^2) -> std deviation of feature j
#
# After computing μ and σ on the training set, use the same values to normalize test data.
# ----------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from multipleVariableLinearRegressionV2 import gradient_descent
# ------------------------------------------------------------------------------
def load_housing_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data/housing_data.txt')
    data = np.loadtxt(data_path, delimiter=',', skiprows = 1)
    x = data[:, :4]  # Features: size, bedrooms, floor, age: selects all rows (:) and first four columns (:4), x becomes an array of shape (m, 4).
    y = data[:, 4] # Target: price: selects all rows and only the 5th column (index 4), y becomes a 1D array of shape (m,)
    return x, y
# ------------------------------------------------------------------------------

x_train, y_train = load_housing_data()
x_features = ['size(sqft)','bedrooms','floors','age']

# ------------------------------------------------------------------------------
# Draw scatter plots for each feature vs price
fig, axs = plt.subplots(1, len(x_features), figsize=(20, 4))
for i, feature in enumerate(x_features):
    axs[i].scatter(x_train[:, i], y_train, alpha=0.5)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('price')
    axs[i].set_title(f'{feature} vs price')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
alpha = 9.9e-7  # Learning rate
alpha = 9e-7
num_iterations = 10  # Number of iterations for gradient descent
# Initialize weights and bias
w_init = np.zeros(x_train.shape[1])  # Initialize weights to zeros, shape (4,)
b_init = 0.0  # Initialize bias to zero
w_final, b_final, j_history, p_history = gradient_descent(
                x_train, 
                y_train, 
                w_init, 
                b_init, 
                alpha, 
                num_iterations)

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations")
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------
#Now lets plot multiple alpha (learning rate) side by side
def evaluate_gradient_descent_with_multiple_learning_rate(x_train, y_train, num_iterations = 100):
    alphas = [9.9e-7, 9e-7, 1e-7]
    w_init = np.zeros(x_train.shape[1])
    b_init = 0.0

    j_histories = []
    w0_histories = []

    for alpha in alphas:
        w, b, j_history, p_history = gradient_descent(
            x_train, y_train, w_init, b_init, alpha, num_iterations
        )
        print(f"alpha={alpha}: final w = {w}, final b = {b}")
        j_histories.append(j_history)
        # If p_history contains weights history, extract w[0] for each iteration
        # Otherwise, you may need to modify gradient_descent to return weights history
        if isinstance(p_history, list) or isinstance(p_history, np.ndarray):
            print("p_history contains weights history, extracting w[0].")
            w0_histories.append([w_iter[0] for w_iter in p_history])
        else:
            print("Warning: p_history does not contain weights history, using fallback.")
            w0_histories.append([w_init[0]] * num_iterations)  # fallback

    fig, axs = plt.subplots(2, len(alphas), figsize=(16, 8))

    for i, (alpha, j_history, w0_history) in enumerate(zip(alphas, j_histories, w0_histories)):
    # Cost vs Iteration
        axs[0, i].plot(j_history)
        axs[0, i].set_title(f"Cost Reduction (alpha={alpha})")
        axs[0, i].set_xlabel("Iteration")
        axs[0, i].set_ylabel("Cost")
        axs[0, i].grid(True)
    # Cost vs w[0]
        axs[1, i].plot(w0_history, j_history, marker='o')
        axs[1, i].set_title(f"Cost vs w[0] (alpha={alpha})")
        axs[1, i].set_xlabel("w[0]")
        axs[1, i].set_ylabel("Cost")
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.show()

evaluate_gradient_descent_with_multiple_learning_rate(x_train, y_train)

# ------------------------------------------------------------------------------
# Now we can implement feature scaling: becase we observed above how gradient scalling is affected by the scale of the features.
# Feature scaling is a technique to standardize the range of independent variables or features of data.
# It helps in speeding up the convergence of gradient descent and improving the performance of the model.

# ------------------------------------------------------------------------------
# Z-Score Normalization (Standardization)
# ---------------------------------------
# This rescales features so that:
#   - The mean (average) becomes 0
#   - The standard deviation becomes 1
# 
# Formula: x_scaled = (x - μ) / σ
# Where:
#   μ = mean of the feature (average of all values)
#   σ = standard deviation (how spread out values are from the mean)
#
# This helps:
#   - Speed up gradient descent convergence
#   - Prevent features with large values from dominating the model
#   - Avoid numerical instability during training
def z_score_normalization(x):
    """
    Apply Z-score normalization to the input features.
    Args:
        x (numpy.ndarray): Input feature matrix of shape (m, n).
    Returns:
        numpy.ndarray: Normalized feature matrix of shape (m, n).
    """
    mu = np.mean(x, axis=0)   # Compute mean for each feature (column)
    sigma = np.std(x, axis=0)  # Compute standard deviation for each feature (column)
    if np.any(sigma == 0):
        raise ValueError("Standard deviation is zero for one or more features, cannot normalize.")  
    x_normalized = (x - mu) / sigma  # Apply Z-score normalization
    return (x_normalized, mu, sigma)

# ------------------------------------------------------------------------------
# Apply Z-score normalization to training data
x_train_normalized, mu, sigma = z_score_normalization(x_train)
# Print the means and standard deviations of the features
print("Means of features:", mu)
print("Standard deviations of features:", sigma)    
# ------------------------------------------------------------------------------
# Now we can run gradient descent on the normalized data

evaluate_gradient_descent_with_multiple_learning_rate(x_train_normalized, y_train, 1000)

# ------------------------------------------------------------------------------