# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# SD: Restrectured ****
# Multivariate Linear Regression for Housing Prices
# ------------------------------------------------------------------------------
# Objective:
# Predict housing prices using multiple features (size, bedrooms, floors, age)
# using multivariate linear regression optimized with gradient descent.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import copy, math
np.set_printoptions(precision=2)  # Set numpy print precision for clarity

# ------------------------------------------------------------------------------
# Step 1: Define Training Data
# ------------------------------------------------------------------------------
x_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print("=== Training Data ===")
print(f"x_train shape: {x_train.shape}, type: {type(x_train)}")
print(x_train)
print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
print(y_train)

# ------------------------------------------------------------------------------
# Step 2: Initialize Parameters (weights and bias)
# ------------------------------------------------------------------------------
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
# These are initial guesses for w and b
print(f"Initial weights: {w_init}, bias: {b_init}")

# ------------------------------------------------------------------------------
# Step 3: Prediction Functions (f_w,b(x) = w·x + b)
# Formula Line Reference: Equation (2) in theory
# ------------------------------------------------------------------------------
def predict_single_loop(x, w, b):
    # f_w,b(x) = sum(w_j * x_j) + b computed manually
    prediction = 0.0
    for i in range(len(w)):
        prediction += w[i] * x[i]  # Corresponds to w_j * x_j
    return prediction + b          # Final addition of b

def predict_vectorized(x, w, b):
    # f_w,b(x) = w · x + b using np.dot
    return np.dot(w, x) + b

# Example prediction check
x_example = x_train[0]
print("\n=== Prediction Example ===")
print("Loop Prediction:", predict_single_loop(x_example, w_init, b_init))
print("Vectorized Prediction:", predict_vectorized(x_example, w_init, b_init))

# ------------------------------------------------------------------------------
# Step 4: Cost Function J(w, b)
# Formula Line Reference: Equation (3)
# ------------------------------------------------------------------------------
def compute_cost(x, y, w, b):
    # J(w,b) = (1 / 2m) * Σ (f_w,b(x^(i)) - y^(i))^2
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(x[i], w) + b         # f_w,b(x^(i))
        cost += (f_wb - y[i]) ** 2         # (f_w,b(x^(i)) - y^(i))^2
    total_cost = (1 / (2 * m)) * cost     # Apply 1/(2m) scaling
    if np.isnan(total_cost) or np.isinf(total_cost):
        raise ValueError("Computed cost is NaN or infinite.")
    return total_cost

print("\n=== Initial Cost ===")
initial_cost = compute_cost(x_train, y_train, w_init, b_init)
print(f"Cost at initial weights: {initial_cost}")

# ------------------------------------------------------------------------------
# Step 5: Gradient Computation ∂J(w,b)/∂w_j and ∂J(w,b)/∂b
# Formula Line Reference: Derivatives from theory
# ∂J/∂w_j = (1/m) * Σ (f_wb(x^(i)) - y^(i)) * x_j^(i)
# ∂J/∂b   = (1/m) * Σ (f_wb(x^(i)) - y^(i))
# ------------------------------------------------------------------------------
def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0
    for i in range(m):
        error = np.dot(x[i], w) + b - y[i]        # error = f_wb(x^(i)) - y^(i)
        for j in range(n):
            try:
                dj_dw[j] += error * x[i, j]           # Accumulate ∂J/∂w_j
            except Exception as e:
                print(f"Unexpected error: {e} at i={i}, x[i]={x[i]}, w={w}, b={b}")
                raise
        dj_db += error                            # Accumulate ∂J/∂b
    dj_dw /= m                                    # Average over m
    dj_db /= m
    if np.isnan(dj_dw).any() or np.isinf(dj_dw).any():
        raise ValueError("NaN/Inf in dj_dw")
    if np.isnan(dj_db) or np.isinf(dj_db):
        raise ValueError("NaN/Inf in dj_db")
    return dj_dw, dj_db

print("\n=== Initial Gradient ===")
grad_w, grad_b = compute_gradient(x_train, y_train, w_init, b_init)
print(f"Gradient w: {grad_w}, Gradient b: {grad_b}")

# ------------------------------------------------------------------------------
# Step 6: Gradient Descent
# Formula Application:
# w_j := w_j - alpha * ∂J/∂w_j
# b   := b - alpha * ∂J/∂b
# ------------------------------------------------------------------------------
def gradient_descent(x, y, w, b, alpha, num_iterations,
                     cost_function=compute_cost,
                     compute_gradient=compute_gradient):
    j_history = []
    p_history = []
    w = copy.deepcopy(w)
    b = float(b)
    for i in range(num_iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)  # Gradient computation
        w -= alpha * dj_dw                           # Update rule for w_j
        b -= alpha * dj_db                           # Update rule for b
        cost = cost_function(x, y, w, b)             # Compute cost J(w, b)
        j_history.append(cost)
        p_history.append((copy.deepcopy(w), float(b)))
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")
    return w, b, j_history, p_history

# ------------------------------------------------------------------------------
# Step 7: Run Training
# ------------------------------------------------------------------------------
alpha = 1e-7
num_iterations = 10000
print("\n=== Running Gradient Descent ===")
w_final, b_final, j_history, p_history = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, num_iterations)

print("\n=== Final Parameters ===")
print(f"w = {w_final}, b = {b_final}")

# ------------------------------------------------------------------------------
# Step 8: Visualization of Cost vs Iteration
# ------------------------------------------------------------------------------
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations")
plt.grid(True)
plt.show()
