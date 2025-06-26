# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Feature Engineering and Polynomial Regression Overview
#
# Linear regression models are built using the formula:
#     f_w,b(x) = w₀x₀ + w₁x₁ + ... + wₙ₋₁xₙ₋₁ + b
# where w represents weights, x are input features, and b is the bias.
#
# This form works well for linearly separable data. However, real-world data like
# housing prices often show non-linear patterns, e.g., pricing may increase with
# living area but flatten out or drop for very large homes.
#
# In such cases, simply tuning w and b won't help. Instead, we need to engineer
# new features (like x², x³, or combinations) to model the non-linear relationships.
# This technique is known as Polynomial Regression and expands our model's capacity.
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from multipleVariableLinearRegressionV2 import gradient_descent
from featureScalingLearningRate import z_score_normalization
np.set_printoptions(precision = 2)  # reduced display precision on numpy arrays

x = np.arange(0, 20, 1 )  # input feature
y = 1 + x**2  # target variable with a quadratic relationship
x = x.reshape(-1, 1)  # reshape x to be a 2D array for compatibility

print("Input feature (x):", x)
print("Target variable (y):", y)

model_w, model_b, j_history, p_history = gradient_descent(x, y, 0, 0, 1e-2, 100)
print("Model weights (w):", model_w)
print("Model bias (b):", model_b)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.plot(x, model_w * x + model_b, label="Predicted Value", color='b') # Predicted value: linear regression: wx + b
plt.title("No Feature Engineering")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# as this is not a great fit, we will use polynomial regression
# ------------------------------------------------------------------------------
# Feature Engineering: Polynomial Regression
# ------------------------------------------------------------------------------
# Well, as expected, not a great fit. What is needed is something like y = w₀ * x₀² + b, 
# or a polynomial feature. To accomplish this, you can modify the input data to *engineer* 
# the needed features. If you swap the original data with a version that squares the x value, 
# then you can achieve y = w₀ * x₀² + b. Let's try it. Swap `x` for `x**2` below:

# Engineer features 
X = x**2      #<-- added engineered feature
X = X.reshape(-1, 1)  #X should be a 2-D Matrix
#model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
model_w, model_b, j_history, p_history = gradient_descent(X, y, 0, 0, 1e-5, 10000)
print("Input feature (x):", X)
print("Target variable (y):", y)


print("Model weights (w) after feature engineering:", model_w)
print("Model bias (b) after feature engineering:", model_b)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); 
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); 
plt.title("Added x**2 feature")
plt.xlabel("x"); 
plt.ylabel("y"); 
plt.legend(); 
plt.show()

# ------------------------------------------------------------------------------
# Insight: Feature Selection via Gradient Descent
#
# Gradient descent emphasizes the features that best fit the target variable.
# Over time, it increases the associated weight (w) of those features while reducing
# the influence of others. For example, in a polynomial model with features like
# x, x², x³, etc., gradient descent may assign a larger weight to x² if it contributes
# most to minimizing the cost.
#
# Gradient descent is effectively 'picking' the correct features by adjusting
# their associated parameters.
#
# Review:
# - Smaller weights imply less important features; if a weight approaches zero,
#   that feature does not help in fitting the model.
# - After training, a large weight on x² (compared to x or x³) indicates it is
#   the most relevant feature for capturing the data pattern.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Let's apply Z-score normalization to our example
# ------------------------------------------------------------------------------
# This will help in scaling the features to have a mean of 0 and a standard deviation of 1.
# This is particularly useful for gradient descent convergence.

x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]  # Create a feature matrix with x, x², and x³
X, mu, sigma = z_score_normalization(X)

print("Normalized Input feature (X):", X)
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")
print(type(X))           # should be <class 'numpy.ndarray'>
print(X.shape)           # should be (20, 3) for example
initial_w = np.zeros(X.shape[1])
initial_b = 0
model_w, model_b, j_history, p_history = gradient_descent(X, y, initial_w, initial_b, 1e-1, 100000)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); 
plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); # x@model_w: performs matrix-vector multiplication 
plt.xlabel("x"); 
plt.ylabel("y"); 
plt.legend(); 
plt.show()
