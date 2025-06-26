# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# ============================================================
# Binary Classification Example - Python Implementation
# ============================================================
#   a basic approach to solving 
#   binary classification problems, such as determining whether:
#       - An email is spam or not
#       - A transaction is fraudulent
#       - A tumor is malignant
# ------------------------------------------------------------------

# responses will always be binary, either 0 or 1
#   - 0: negative class (e.g., not spam, not fraudulent, benign tumor)
#   - 1: positive class (e.g., spam, fraudulent, malignant tumor)

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9 ])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Binary responses

plt.scatter(x_train, y_train, c=y_train, cmap='bwr', edgecolors='k')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.title('Binary Classification Data')
plt.show()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------
# Sigmoid (Logistic) Function for Classification
# ------------------------------------------------------------------
# In binary classification, we want the model's output to be between 0 and 1,
# so that it can represent a probability that a sample belongs to the positive class.
# 
# This is achieved using the **sigmoid function**, defined as:
#       g(z) = 1 / (1 + e^(-z))
# 
# Where:
#   - z is the input to the function (typically the output of a linear model)
#   - g(z) is the output, which maps any real value into (0, 1)
#
# Properties:
#   - g(z) → 0 as z → -∞
#   - g(z) → 1 as z → +∞
#   - g(0) = 0.5
#
# This makes it ideal for turning the linear output of a model into a 
# probability suitable for classification.
#
# NumPy’s 'np.exp()' function is typically used to compute the exponential part.

input_array = np.array([1,2,3])  
exp_array = np.exp(input_array)  # Compute e^(-z)

print("Input array:", input_array)
print("Exponential of input_array:", exp_array)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# ------------------------------------------------------------------------------

# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Apply the sigmoid function to each value in the array
sigmoid_values = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, sigmoid_values])

# Plotting the sigmoid function
plt.figure(figsize=(10, 6))
plt.plot(z_tmp, sigmoid_values, label='Sigmoid Function', color='blue')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('g(z)')
plt.axhline(0.5, color='red', linestyle='--', label='g(z) = 0.5')
plt.axvline(0, color='green', linestyle='--', label='z = 0')
plt.legend()
plt.grid(True)
plt.show()
# ------------------------------------------------------------------------------

# As you can see, the sigmoid function approaches 0 as z goes to large negative values and approaches 1 as z goes to large positive values.
# This makes it suitable for binary classification tasks, where we want to predict the probability of a sample belonging to the positive class.
