import numpy as np
import matplotlib.pyplot as plt
# ===========================================
# Function to Generate Synthetic Coffee Data
# ===========================================
def load_coffee_data():
    """
    Creates a synthetic coffee roasting dataset.
    Optimal roasting conditions:
        - Duration: 12-15 minutes
        - Temperature: 175-260C
    Returns:
        X: [temperature, duration]
        Y: label (1=good roast, 0=bad roast)
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5              # Duration range ~11.5 to 15.5
    X[:, 0] = X[:, 0] * (285 - 150) + 150     # Temperature range ~150 to 285

    Y = np.zeros(len(X))
    for i, (t, d) in enumerate(X):
        y_boundary = -3 / (260 - 175) * t + 21  # Linear decision boundary
        if (175 < t < 260 and 12 < d < 15 and d <= y_boundary):
            Y[i] = 1
    return X, Y.reshape(-1, 1)
# ===========================