# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Gradient Descent for Linear Regression (One Feature)
#
# Linear Model Prediction:
#     f_wb(x[i]) = w * x[i] + b
#
# Cost Function:
#     J(w, b) = (1 / 2m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i])²
#
# Gradient Descent Update Rule:
#     repeat until convergence:
#         w = w - α * ∂J(w, b)/∂w
#         b = b - α * ∂J(w, b)/∂b
#     (α is the learning rate)
#
# Gradients:
#     ∂J(w, b)/∂w = (1 / m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i]) * x[i]
#     ∂J(w, b)/∂b = (1 / m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i])
#
# Notes:
# - f_wb(x[i]) is the model's prediction for the i-th training example.
# - You calculate gradients for all parameters *before* updating any of them.
# - These updates move the parameters w and b in the direction that minimizes J(w, b).
#
# You will need to implement:
#   1. compute_cost      -> uses the cost function (J)
#   2. compute_gradient  -> uses the partial derivative formulas above
#   3. gradient_descent  -> uses both to update w and b iteratively
#
# Conventions:
# - Naming of gradient variables should follow the format:
#     dj_dw for ∂J(w, b)/∂w
#     dj_db for ∂J(w, b)/∂b
# ------------------------------------------------------------------------------
# Not ideal - not converging at the end, but it is a good start.
# ------------------------------------------------------------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
from costFunction import compute_cost

x_train = np.array([1.0, 2.0, 3, 5, 8, 10])  # Input variable (size in 1000 square feet)
y_train = np.array([300.0, 500.0, 600, 700, 800, 1000])  # Target (price in 1000s of dollars)

def compute_gradient(x, y, w, b):
    """
    Computes the gradients of the cost function with respect to w and b.
    
    Args:
        x (ndarray (m,)): Input data, m examples
        y (ndarray (m,)): Target values
        w (scalar): Weight parameter
        b (scalar): Bias parameter
    
    Returns:
        tuple: Gradients (dj_dw, dj_db)
    """
    m = x.shape[0]  # Number of training examples
    if m == 0:
        raise ValueError("Input array x is empty.")
    if not isinstance(w, (int, float)):
        raise TypeError("Weight w must be a number.")
    if not isinstance(b, (int, float)):
        raise TypeError("Bias b must be a number.")
    if x.ndim != 1:
        raise ValueError("Input x must be a 1-dimensional array.")
    
    dj_dw = 0.0  # Initialize gradient for w
    dj_db = 0.0  # Initialize gradient for b
    
    # ∂J(w, b)/∂w = (1 / m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i]) * x[i]
    # ∂J(w, b)/∂b = (1 / m) * Σ[i=0 to m-1] (f_wb(x[i]) - y[i])
    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error
    
    # Average the gradients
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

# example usage
w = 0.0  # Initial weight
b = 0.0  # Initial bias

gradientVal = compute_gradient(x_train, y_train, w, b)
print(f"Gradient w: {gradientVal[0]}, Gradient b: {gradientVal[1]}")

# Gradient Descent Function
# ------------------------------------------------------------------------------
# Gradient Descent for Linear Regression (One Feature)
#     repeat until convergence:
#         w = w - α * ∂J(w, b)/∂w
#         b = b - α * ∂J(w, b)/∂b
#     (α is the learning rate)
# ------------------------------------------------------------------------------

def gradient_descent(x, y, w, b, alpha, num_iterations, cost_function = compute_cost, compute_gradient = compute_gradient):
    # returnJ_history (List): History of cost values
    # p_history (list): History of parameters [w,b] 
    """
    Performs gradient descent to learn parameters w and b.
    Args:
        x (ndarray (m,)): Input data, m examples
        y (ndarray (m,)): Target values
        w (scalar): Initial weight parameter
        b (scalar): Initial bias parameter
        alpha (float): Learning rate
        num_iterations (int): Number of iterations for gradient descent
        cost_function (function): Function to compute cost
        compute_gradient (function): Function to compute gradients
    Returns:
        tuple: Final parameters (w, b), cost history, and parameter history
    """
    j_history = []  # List to store cost history
    p_history = []  # List to store parameter history

    for i in range(num_iterations):
        # Compute gradients
        dj_dw, dj_db = compute_gradient(x, y, w, b) #∂J(w, b)/∂w and ∂J(w, b)/∂b

        # Update parameters
        w = w - alpha * dj_dw # w = w - α * ∂J(w, b) / ∂w
        b = b - alpha * dj_db # b = b - α * ∂J(w, b) / ∂b

        if i < 100000:
            # Compute cost and store it
            j = cost_function(x, y, w, b)
            j_history.append(j)

            # Store parameters
            p_history.append((w, b))

        if i% math.ceil(num_iterations / 10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, j_history, p_history 

# Example usage
alpha = 0.001  # Learning rate
num_iterations = 100000  # Number of iterations for gradient descent
w_final, b_final, j_history, p_history = gradient_descent(x_train, y_train, w, b, alpha, num_iterations)
print(f"Final cost after gradient descent: {j_history[-1]}")
# Plotting the cost history
plt.plot(j_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.grid()
plt.show()
# Plotting the parameter history
plt.plot([p[0] for p in p_history], label='Weight (w)')
plt.plot([p[1] for p in p_history], label='Bias (b)')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameter History during Gradient Descent')
plt.legend()
plt.grid()
plt.show()

plt.scatter(x_train, y_train, label='Data')
y_pred = w_final * x_train + b_final
plt.plot(x_train, y_pred, color='red', label='Prediction')
plt.legend()
plt.title("Model Prediction vs Data")
plt.show()

# Final parameters after gradient descent

print(f"Final parameters after gradient descent: w = {w_final}, b = {b_final}")
print(f"Final weight: {w_final}, Final bias: {b_final}")
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")