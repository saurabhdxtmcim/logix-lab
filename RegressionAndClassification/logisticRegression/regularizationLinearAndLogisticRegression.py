# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 📉 Cost Function with Regularization (Logistic or Linear Regression)
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
from sigmoid import sigmoid  

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

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_regression(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

# ------------------------------------------------------------------------------
# Regularized Logistic Regression Cost Function
# ------------------------------------------------------------------------------
# Regularization helps prevent overfitting by discouraging large weight values.
# It does this by adding a penalty term to the cost function.
# The total cost function is made up of two parts:
# 1. Logistic loss (same as before):
#     J(w, b) = (1 / m) * Σ [ -y * log(f(x)) - (1 - y) * log(1 - f(x)) ]
#
#     Where:
#       - m = number of training examples
#       - f(x) = sigmoid(w · x + b) → predicted probability

# 2️. Regularization term (penalty on large weights):
#     (λ / 2m) * Σ (wⱼ²)
#
#     Where:
#       - λ is the regularization strength (lambda)
#       - This term **does NOT include the bias b**

# So the **final cost function** becomes:
#     J(w, b) = [logistic loss] + [regularization penalty]

#  Why do this?
# - Helps avoid overfitting by keeping weights small
# - Encourages simpler models that generalize better

def compute_cost_logistic_regression(x, y, w, b, lambda_=1.0):
    """    
    Computes the regularized cost for logistic regression.
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
        f_wb_i = sigmoid(z_i)           # f_wb⁽ⁱ⁾ = sigmoid(z⁽ⁱ⁾) → predicted probability
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i) # Add logistic loss: -y * log(f_wb) - (1 - y) * log(1 - f_wb)
    # Average cost over all examples: (1 / m) * Σ logistic loss
    cost /= m                                  # (1 / m) * Σ logistic loss → unregularized cost
    # Regularization term: (λ / 2m) * Σ(wⱼ²)
    reg_term = (lambda_ / (2 * m)) * np.sum(w ** 2)  # Excludes bias term b
    return cost + reg_term                     # Total cost = unregularized cost + regularization

np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_regression(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

# ------------------------------------------------------------------------------
#  Gradient Descent with Regularization (Linear/Logistic Regression)
# ------------------------------------------------------------------------------

# The update rule for gradient descent **does not change** with regularization.
# You still update weights and bias like this:
#
#     w_j = w_j - α * ∂J/∂w_j
#     b   = b   - α * ∂J/∂b
#
# The only difference is: how we compute the gradients (∂J/∂w_j and ∂J/∂b)

# ----------------------------------------------------------------------
# For both Linear and Logistic Regression, the gradients become:
#
# ∂J/∂w_j = (1 / m) * Σ (f_wb(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x_j⁽ⁱ⁾   +   (λ / m) * w_j
# ∂J/∂b   = (1 / m) * Σ (f_wb(x⁽ⁱ⁾) - y⁽ⁱ⁾)
#
#  The extra term (λ / m) * w_j is the **regularization penalty**
#     - It "pushes" weights toward zero to reduce overfitting
#     - Note: bias `b` is NOT regularized (only weights `w`)

# ----------------------------------------------------------------------
#  Model predictions f_wb(x⁽ⁱ⁾) differ by type of regression:
# - For Linear Regression:     f(x) = w · x + b
# - For Logistic Regression:   f(x) = sigmoid(w · x + b)
# Regularization makes gradient descent favor **simpler models** by discouraging large weights.
# --------------------------------------------------------------------------

# Gradient function for regularized linear regression
def compute_gradient_linear_regression(x, y, w, b, lambda_=1.0):
    """
    Computes the gradients of the regularized cost function for linear regression.
    Args:
        x: Training features, shape (m, n)
        y: Labels, shape (m,)
        w: Weights, shape (n,)
        b: Bias term (scalar)
        lambda_: Regularization parameter (default is 1.0)
    Returns:
        dj_dw: Gradient with respect to weights, shape (n,)
        dj_db: Gradient with respect to bias (scalar)
    """
    m, n = x.shape                             # m = number of examples, n = number of features
    dj_dw = np.zeros(n, dtype=np.longdouble)   # ∂J/∂w initialized to zeros
    dj_db = 0.0                                 # ∂J/∂b initialized to 0
    for i in range(m):                          # Loop over training examples
        z_i = np.dot(x[i], w) + b               # z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b
        f_wb_i = z_i                            # f_wb⁽ⁱ⁾ = z⁽ⁱ⁾ for linear regression
        error = f_wb_i - y[i]                   # (f_wb - y) part of the gradient formula
        dj_dw += error * x[i]                   # Accumulate ∂J/∂w: add (f_wb - y) * x⁽ⁱ⁾
        dj_db += error                          # Accumulate ∂J/∂b: add (f_wb - y)  
    dj_dw /= m                                  # Final average: (1/m) * Σ[∂J/∂w]
    dj_db /= m                                  # Final average: (1/m) * Σ[∂J/∂b]
    # Add regularization term: (λ / m) * w
    dj_dw += (lambda_ / m) * w                 # ∂J/∂w with regularization
    return dj_dw, dj_db                         # Return the gradients

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_regression(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

#Gradient function for regularized logistic regression
def compute_gradient_logistic_regression(x, y, w, b, lambda_=1.0):
    """
    Computes the gradients of the regularized cost function for logistic regression.
    Args:
        x: Training features, shape (m, n)
        y: Labels, shape (m,)
        w: Weights, shape (n,)
        b: Bias term (scalar)
        lambda_: Regularization parameter (default is 1.0)
    Returns:
        dj_dw: Gradient with respect to weights, shape (n,)
        dj_db: Gradient with respect to bias (scalar)
    """
    m, n = x.shape                             # m = number of examples, n = number of features
    dj_dw = np.zeros(n, dtype=np.longdouble)   # ∂J/∂w initialized to zeros
    dj_db = 0.0                                 # ∂J/∂b initialized to
    for i in range(m):                          # Loop over training examples
        z_i = np.dot(x[i], w) + b               # z⁽ⁱ⁾ = w · x⁽ⁱ⁾ + b
        f_wb_i = sigmoid(z_i)           # f_wb⁽ⁱ⁾ = sigmoid(z⁽ⁱ⁾) → predicted probability
        error = f_wb_i - y[i]                   # (f_wb - y) part of the gradient formula
        dj_dw += error * x[i]                   # Accumulate ∂J/∂w: add (f_wb - y) * x⁽ⁱ⁾
        dj_db += error                          # Accumulate ∂J/∂b: add (f_wb - y)
    dj_dw /= m                                  # Final average: (1/m) * Σ[∂J/∂w]   
    dj_db /= m                                  # Final average: (1/m) * Σ[∂J/∂b]
    # Add regularization term: (λ / m) * w
    dj_dw += (lambda_ / m) * w                 # ∂J/∂w with regularization
    return dj_dw, dj_db                         # Return the gradients

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_regression(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )