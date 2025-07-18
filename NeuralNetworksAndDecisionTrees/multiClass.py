# ------------------------------------------
# Multi-class Classification - Neural Networks
# ------------------------------------------
#
# Neural Networks are often used to classify data into multiple categories.
# Examples:
#   - Classifying objects in photos: {dog, cat, horse, other}
#   - Classifying parts of speech in a sentence: {noun, verb, adjective, ...}
#
# A multiclass network has:
#   - Multiple outputs in its last layer (one per class/category)
#   - The largest output determines the predicted class
#   - A softmax function converts outputs to class probabilities
#
# In this example, we:
#   - Build and train a 2-layer network on a 4-class synthetic dataset
#   - Visualize predictions and decision boundaries
#   - Show how each layer transforms data
# ------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress TensorFlow info/warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.datasets import make_blobs

np.set_printoptions(precision=2)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Number of classes and dataset parameters
classes = 4
m = 100
centers = np.array([
    [-5, 2],   # Class 0
    [-2, -2],  # Class 1
    [2, 2],    # Class 2
    [5, -2]    # Class 3
])
std = 1.0

# --- Create a synthetic dataset with 4 classes ---
X, y = make_blobs(n_samples=1000, centers=centers, random_state=42, cluster_std=std)

# --- Plot the dataset ---
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=5)
plt.title("Synthetic Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar()
plt.show()

# --- Print dataset summary ---
print(f"unique classes: {np.unique(y)}"
      f"\nnumber of samples: {len(y)}"
      f"\nnumber of features: {X.shape[1]}")
print(f"class representation:\n{np.bincount(y)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# --- Build and train the neural network ---
tf.random.set_seed(1234)  # For reproducibility

model = Sequential([
    Dense(32, activation='relu', name="L1"),    # First hidden layer, 32 units
    Dense(4, activation='linear', name="L2")    # Output layer, 4 units (one per class), no softmax!
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# --- Extract weights and biases of both layers ---
l1 = model.get_layer("L1")
w1, b1 = l1.get_weights()
print(f"Layer 1 weights shape: {w1.shape}\nLayer 1 weights (first 5):\n{w1[:5]}")
print(f"Layer 1 biases shape: {b1.shape}\nLayer 1 biases (first 5):\n{b1[:5]}")

l2 = model.get_layer("L2")
w2, b2 = l2.get_weights()
print(f"Layer 2 weights shape: {w2.shape}\nLayer 2 weights (first 5):\n{w2[:5]}")
print(f"Layer 2 biases shape: {b2.shape}\nLayer 2 biases:\n{b2}")

# --- Helper function to plot the model's decision boundary ---
def plot_decision_boundary(model, X, ax, title="Decision Boundary"):
    """
    Plots the decision boundary for a given trained model on X.

    Parameters:
        model -- Trained Keras model
        X -- Dataset for plotting limits
        ax -- Matplotlib axes to draw on
        title -- Plot title
    """
    # Create a mesh grid covering the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Predict probabilities for every grid point, then choose the class with highest probability
    logits = model.predict(grid, verbose=0)
    Z = np.argmax(logits, axis=1)
    Z = Z.reshape(xx.shape)
    # Plot decision boundary as color regions
    contour = ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    return contour

# --- Plot the model's decision boundary in original feature space ---
fig, ax = plt.subplots(figsize=(8, 6))
# Scatter plot colored by predicted class
scatter = ax.scatter(X[:, 0], X[:, 1], c=model.predict(X, verbose=0).argmax(axis=1), cmap='viridis', s=5)
plot_decision_boundary(model, X, ax, title="Predicted Classes after Output Layer")
ax.set_title("Predicted Classes after Output Layer")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.colorbar(scatter, ax=ax)
plt.show()

# --- Extract the "new features" created by the first hidden layer (L1) ---
# This is the output after L1, before the final linear layer
Xl1 = np.maximum(0, np.dot(X, w1) + b1)  # Apply ReLU

# --- Define function that applies only the final layer (L2) to features after L1 ---
def layer2_only(X_transformed):
    """
    Forward pass through only the second layer (L2) using features after L1 (ReLU).
    """
    # X_transformed shape: [n_samples, 32] (must match w2 shape)
    logits = np.dot(X_transformed, w2) + b2
    return logits

# --- Helper to plot decision boundary in the L1 feature space ---
def plot_decision_boundary_l2(Xl1, ax, title="Decision Boundary after Layer 2"):
    """
    Plots decision boundary after L2, operating directly on L1's transformed features.
    """
    x_min, x_max = Xl1[:, 0].min() - 1, Xl1[:, 0].max() + 1
    y_min, y_max = Xl1[:, 1].min() - 1, Xl1[:, 1].max() + 1
    # Only plot the first two "new features" (for visualization); rest are ignored for plotting
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Pad grid to 32 dimensions with zeros, so shape matches w2
    n_extra = w2.shape[0] - 2
    if n_extra > 0:
        grid = np.hstack([grid, np.zeros((grid.shape[0], n_extra))])
    # Now grid shape is [num_points, 32]
    logits = layer2_only(grid)
    Z = np.argmax(logits, axis=1)
    Z = Z.reshape(xx.shape)
    contour = ax.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')
    return contour

# --- Visualize data in the new feature space after L1 transformation ---
fig, ax = plt.subplots(figsize=(8, 6))
# Scatter plot, coloring by the model's predictions (original input X, but colored by model output)
scatter = ax.scatter(Xl1[:, 0], Xl1[:, 1], c=model.predict(X, verbose=0).argmax(axis=1), cmap='viridis', s=5)
plot_decision_boundary_l2(Xl1, ax, title="New Features after Layer 2")
ax.set_title("Decision Boundary in L1 Feature Space (First Two Dimensions)")
ax.set_xlabel("First L1 Feature")
ax.set_ylabel("Second L1 Feature")
plt.colorbar(scatter, ax=ax)
plt.show()


