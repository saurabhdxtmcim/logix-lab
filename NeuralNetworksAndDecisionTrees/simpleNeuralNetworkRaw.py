# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

# Suppress TensorFlow debugging messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Import utility functions
from coffeeData import load_coffee_data  # Function to load the coffee roasting dataset
from sigmoid import sigmoid              # Sigmoid activation function

# ==================================================
# Load and Visualize the Raw Coffee Roasting Dataset
# ==================================================
x, y = load_coffee_data()  # x: features, y: labels
print("Raw data shape:", x.shape, y.shape)

# Scatter plot of the data: temperature vs. duration colored by quality
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=10)
plt.xlabel('Temperature (C)')
plt.ylabel('Roasting Duration (min)')
plt.title('Coffee Roasting Data')
plt.colorbar(label='Quality (0=bad, 1=good)')
plt.show()

# ===================================
# Normalize the Features for Training
# ===================================
print(f"Temperature Max, Min pre normalization: {np.max(x[:,0]):0.2f}, {np.min(x[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(x[:,1]):0.2f}, {np.min(x[:,1]):0.2f}")

# Create a normalization layer to standardize the data
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(x)  # Compute mean and variance for normalization

# Normalize input features
xn = norm_l(x)

# Display post-normalization stats
print(f"Temperature Max, Min post normalization: {np.max(xn[:,0]):0.2f}, {np.min(xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(xn[:,1]):0.2f}, {np.min(xn[:,1]):0.2f}")

# ========================================
# Define Dense Layer and Neural Net Logic
# ========================================

# Alias for sigmoid activation
g = sigmoid

def my_dense(a_in, W, b):
    """
    Simulates a single dense layer using NumPy.
    
    Parameters:
        a_in -- input activations (1D vector)
        W    -- weight matrix (shape: [n_inputs, n_units])
        b    -- bias vector (shape: [n_units])
    
    Returns:
        a_out -- output activations after applying weights, bias, and sigmoid
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(W[:, j], a_in) + b[j]  # Linear combination
        a_out[j] = g(z)                   # Activation
    return a_out

def my_sequential(x, W1, b1, W2, b2):
    """
    Forward pass through a 2-layer neural network.

    Parameters:
        x   -- input features (1D vector)
        W1  -- weights for layer 1
        b1  -- biases for layer 1
        W2  -- weights for layer 2
        b2  -- biases for layer 2

    Returns:
        a_out -- final output after 2 layers
    """
    a1 = my_dense(x, W1, b1)         # First hidden layer
    a_out = my_dense(a1, W2, b2)     # Output layer
    return a_out

# ==================================================
# Set Pre-trained Weights and Biases for Prediction
# ==================================================
# These weights and biases are pre-computed from a trained model
W1_tmp = np.array([[ -8.93,  0.29, 12.90],
                   [ -0.10, -7.32, 10.81]])   # Shape: [2, 3] => 2 inputs, 3 neurons
b1_tmp = np.array([ -9.82, -9.28,  0.96])     # Shape: [3]

W2_tmp = np.array([[-31.18],
                   [-27.59],
                   [-32.56]])                # Shape: [3, 1] => 3 inputs from previous layer, 1 output neuron
b2_tmp = np.array([15.41])                   # Shape: [1]

# ==================================================
# Make Predictions on New Data
# ==================================================
def my_predict(X, W1, b1, W2, b2):
    """
    Predicts outputs for a batch of inputs using the 2-layer model.

    Parameters:
        X  -- matrix of input samples (shape: [n_samples, n_features])
    
    Returns:
        p -- predictions (shape: [n_samples, 1])
    """
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)
    return p

# Two new roasting settings to classify
X_tst = np.array([
    [200, 13.9],  # Expected to be "good" roast
    [200, 17.0]   # Expected to be "bad" roast
])

# Normalize the test inputs using the same stats as training
X_tstn = norm_l(X_tst)

# Run predictions
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

# Threshold predictions to convert to class labels
yhat = (predictions >= 0.5).astype(int)
print(f"Decisions (0=bad, 1=good):\n{yhat}")

# ========================================
# Visualize Predictions on the Input Space
# ========================================
plt.figure(figsize=(10, 6))
# Plot training data
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=10, label='Training Data')
# Plot test data points
plt.scatter(X_tst[:, 0], X_tst[:, 1], c='black', marker='x', s=100, label='Test Data')
# Overlay model predictions on the test points
plt.scatter(X_tst[:, 0], X_tst[:, 1], c=yhat.flatten(), cmap='bwr', alpha=0.5, label='Predictions')
plt.title("Coffee Roast Classification: Model Predictions vs Data")
plt.xlabel("Temperature (C)")
plt.ylabel("Roasting Duration (min)")
plt.colorbar(label='Quality (0=bad, 1=good)')
plt.legend()
plt.show()
# ==================================================