import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import logging
from coffeeData import load_coffee_data  # Import the coffee data generation function

# Suppress TensorFlow logging for clarity
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# ===========================
# Visualize the Raw Dataset
# ===========================
x, y = load_coffee_data()
print("Raw data shape:", x.shape, y.shape)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', s=10)
plt.xlabel('Temperature (C)')
plt.ylabel('Roasting Duration (min)')
plt.title('Coffee Roasting Data')
plt.colorbar(label='Quality (0=bad, 1=good)')
plt.show()

# ================================
# Normalize Data for Training
# ================================
# Using a Keras Normalization Layer:
# 1. Create the normalization layer (not a model layer here)
# 2. Adapt it to learn mean and variance of the training data
# 3. Apply it to normalize input features
print("\nNormalizing data...")
normalization_layer = tf.keras.layers.Normalization(axis=-1)
normalization_layer.adapt(x)

xn = normalization_layer(x)

print("Normalization layer mean:", normalization_layer.mean.numpy())
print("Normalization layer variance:", normalization_layer.variance.numpy())

# Optional: Print value ranges post-normalization
print(f"Temperature range post-norm: {np.min(xn[:,0]):.2f} to {np.max(xn[:,0]):.2f}")
print(f"Duration range post-norm: {np.min(xn[:,1]):.2f} to {np.max(xn[:,1]):.2f}")

# ===========================
# Expand Data for Training
# ===========================
xt = np.tile(xn, (1000, 1))   # Repeat input data
yt = np.tile(y, (1000, 1))    # Repeat labels
print("Training data shape:", xt.shape, yt.shape)

# ================================
# Build the Coffee Roasting Model
# ================================
# A simple feedforward neural network with 3 layers
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(64, activation='sigmoid', name='layer1'),
    Dense(32, activation='sigmoid', name='layer2'),
    Dense(1, activation='sigmoid', name='output')
])

model.summary()

# ===========================
# Compile and Train the Model
# ===========================
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)

# ================================
# Epochs and Batches (Training)
# ================================
# - `epochs=10` means the model sees all training data 10 times.
# - `batch_size=32`: data is divided into batches of 32 samples.
# - `validation_split=0.2`: 20% used to validate during training.
# - `verbose=1`: enables training progress bar.
model.fit(
    xt,
    yt,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =============================
# View Trained Model Weights
# =============================
for layer_name in ["layer1", "layer2", "output"]:
    W, b = model.get_layer(layer_name).get_weights()
    print(f"\nWeights of {layer_name}:\nW shape: {W.shape}\n", W, f"\nb shape: {b.shape}\n", b)

# =============================
# Make Predictions on Test Data
# =============================
x_test = np.array([
    [200, 13.9],  # Should be a "good" roast
    [200, 17]     # Should be a "bad" roast (over-roasted)
])
x_testn = normalization_layer(x_test)

predictions = model.predict(x_testn)
print("\nPredictions (probabilities):\n", predictions)

# Convert predicted probabilities to binary decisions
# Threshold of 0.5 is used
print("\nDecisions (0 = bad, 1 = good):")
yhat = (predictions >= 0.5).astype(int)
print(yhat)
