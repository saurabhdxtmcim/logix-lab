# -----------------------------------------------
# TensorFlow and Keras Setup
# -----------------------------------------------
# TensorFlow is a machine learning library developed by Google.
# Keras is a high-level API for building and training models.
# Since TensorFlow 2.0, Keras is integrated directly into TensorFlow.

import numpy as np                              # Import NumPy for numerical operations
import matplotlib.pyplot as plt                  # Import Matplotlib for plotting
import tensorflow as tf                          # Import TensorFlow core library
from tensorflow.keras.layers import Dense, Input # Import layers for model building
from tensorflow.keras import Sequential          # Import Sequential model API
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy # Loss functions
from tensorflow.keras.activations import sigmoid # Sigmoid activation
import logging                                   # For suppressing TensorFlow logs
from sigmoid import sigmoid                      # Custom sigmoid function (if applicable)

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# -----------------------------------------------
# Linear Regression Model (No Activation)
# -----------------------------------------------
# Regression / Linear Model Concept:
# -----------------------------------------------
# A neuron without an activation function implements
# the same behavior as a linear regression model.
#
# This is described by the function:
#     f_w,b(x^(i)) = w · x^(i) + b
#
# In this case, a single neuron (or unit) behaves just
# like the familiar linear regression model from Course 1.
#
# This concept helps bridge understanding between classical
# machine learning and neural network layers.
# -----------------------------------------------

# Input and output training data
x_train = np.array([[1.0], [2.0]], dtype=np.float32)  # Input: size in 1000 sq ft
y_train = np.array([[300.0], [500.0]], dtype=np.float32)  # Output: price in $1000s

# Plot training data
fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'ro', label='Training data')
ax.set_xlabel('Size (1000 sq ft)')
ax.set_ylabel('Price (1000s of dollars)')
ax.set_title('Training Data')
ax.legend()
plt.show()

# Create a single neuron with linear activation (i.e., no activation)
linear_layer = Dense(units=1, activation='linear')

# Check initialized weights and biases (random by default)
linear_layer.get_weights()

# Perform a forward pass with the first input sample
a1 = linear_layer(x_train[0].reshape(1, 1))
print(a1)  # Output from the neuron

# Retrieve weights and bias after initialization
w, b = linear_layer.get_weights()
print(f'Weights: {w}, Bias: {b}')

# Manually set weights and bias
set_w = np.array([[200.0]], dtype=np.float32)
set_b = np.array([100.0], dtype=np.float32)
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())  # Confirm set weights and bias

# Forward pass again with updated weights
a2 = linear_layer(x_train[0].reshape(1, 1))
print(a2)  # Should be 200*1 + 100 = 400

# Make predictions for all inputs
prediction_tf = linear_layer(x_train)  # TensorFlow output
prediction_np = np.dot(x_train, set_w) + set_b  # NumPy calculation
print(prediction_tf)  # Expected: [400.0, 500.0]
print(prediction_np)

# Plot model prediction vs training data
plt.plot(x_train, prediction_tf, 'b-', label='Model Prediction')
plt.plot(x_train, y_train, 'ro', label='Training Data')
plt.xlabel('Size (1000 sq ft)')
plt.ylabel('Price (1000s of dollars)')
plt.title('Model Prediction vs Training Data')
plt.legend()
plt.show()

# -----------------------------------------------
# Logistic Regression Model (Sigmoid Activation)
# -----------------------------------------------
# A 'logistic neuron' is created by applying a sigmoid
# activation to the output of a neuron. This implements
# the logistic regression function.
# A neuron or unit with a sigmoid activation function
# behaves like a logistic regression model.
#
# This is expressed as:
#     f_w,b(x^(i)) = g(w · x^(i) + b)
# where
#     g(x) = sigmoid(x)
#
# This transforms the linear combination of inputs
# into a value between 0 and 1, making it ideal for
# binary classification problems.
#
# We'll now assign values to weights (w) and bias (b)
# and observe the model's behavior.
# -----------------------------------------------

# Redefine training data for binary classification
x_train = np.array([0., 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape(-1, 1)
y_train = np.array([0., 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

# Create masks for plotting positive and negative classes
pos = y_train == 1
neg = y_train == 0

# Plot training data for classification
fig, ax = plt.subplots()
ax.plot(x_train[pos], y_train[pos], 'ro', label='Positive Class')
ax.plot(x_train[neg], y_train[neg], 'bo', label='Negative Class')
ax.set_xlabel('Feature Value')
ax.set_ylabel('Target Value')
ax.set_title('Training Data')
ax.legend()
plt.show()

# Build logistic regression model using TensorFlow's Keras Sequential API
model = Sequential([
    Dense(1, input_dim=1, activation='sigmoid', name='logistic_unit')
])

# Display model architecture
model.summary()

# Access weights and bias from the logistic unit
logistic_layer = model.get_layer('logistic_unit')
w, b = logistic_layer.get_weights()
print(f'Weights: {w}, Bias: {b}')

# Manually set weights and bias for predictable output
set_w = np.array([[2.0]], dtype=np.float32)
set_b = np.array([-10.5], dtype=np.float32)
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

# Perform prediction on first sample using the model
a1 = model.predict(x_train[0].reshape(1, 1))
print(a1)  # Should be close to sigmoid(0*2 - 4.5) ~ 0.011

# Confirm with NumPy using custom sigmoid
alog = sigmoid(np.dot(x_train[0], set_w) + set_b)
print(alog)

# Plot model predictions against the true labels
plt.plot(x_train, model.predict(x_train), 'b-', label='Model Prediction')
plt.plot(x_train[pos], y_train[pos], 'ro', label='Positive Class')
plt.plot(x_train[neg], y_train[neg], 'bo', label='Negative Class')
ax.set_xlabel('Feature Value')
ax.set_ylabel('Target Value')
ax.set_title('Model Prediction vs Training Data')
ax.legend()
plt.show()

# -----------------------------------------------
# Summary
# -----------------------------------------------
# In this section, we explored how a single neuron can
# be used to model both linear and logistic regression tasks.
# We implemented a simple linear regression model without
# activation and a logistic regression model with a sigmoid
# activation function. Through training and evaluation, we
# observed how the model learns to separate classes and make
# predictions.