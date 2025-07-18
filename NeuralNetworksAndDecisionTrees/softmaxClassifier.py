# softmaxClassifier.py
#  it is a Reusable softmax classifier for multiclass classification
#  using TensorFlow/Keras with a numerically stable setup.
# ------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Optional, List, Any, Union
import os

class SoftmaxClassifier:
    """
    A production-grade, reusable softmax classifier for multiclass classification
    using TensorFlow/Keras with a numerically stable setup.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        model_dir: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the SoftmaxClassifier.

        Parameters:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            hidden_layers (List[int], optional): Sizes of hidden layers.
            learning_rate (float): Learning rate for the optimizer.
            model_dir (str, optional): Directory to save/load the model.
            random_seed (int, optional): Random seed for reproducibility.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers or [25, 15]
        self.learning_rate = learning_rate
        self.model_dir = model_dir or "./softmax_model"
        self.model: Optional[Sequential] = None

        self._build_model()

    def _build_model(self):
        """Build and compile the internal Keras model."""
        model = Sequential()
        for units in self.hidden_layers:
            model.add(Dense(units, activation='relu', input_dim=self.input_dim if model.layers == [] else None))
        model.add(Dense(self.num_classes, activation='linear'))  # No softmax!
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        self.model = model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.1,
        patience: int = 5,
        checkpoint: bool = True,
        verbose: int = 1
    ):
        """
        Train the model with optional early stopping and checkpointing.

        Parameters:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            validation_split (float): Fraction for validation split.
            patience (int): Patience for early stopping.
            checkpoint (bool): Whether to save the best model.
            verbose (int): Keras verbosity level.
        """
        callbacks = []
        if patience > 0:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
        if checkpoint:
            os.makedirs(self.model_dir, exist_ok=True)
            ckpt_path = os.path.join(self.model_dir, "best_model.keras")
            callbacks.append(ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss'))

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        return history

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Predict raw logits for input X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Logits (shape: [num_samples, num_classes])
        """
        self._check_model()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict softmax probabilities for input X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        logits = self.predict_logits(X)
        return tf.nn.softmax(logits, axis=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for input X.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, verbose: int = 1) -> dict:
        """
        Evaluate the model on test data.

        Parameters:
            X (np.ndarray): Test features.
            y (np.ndarray): True labels.
            batch_size (int): Batch size.
            verbose (int): Verbosity.

        Returns:
            dict: Dictionary with loss and metrics.
        """
        self._check_model()
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)
        metrics = dict(zip(self.model.metrics_names, results))
        return metrics

    def save(self, path: Optional[str] = None):
        """
        Save the trained model to disk.

        Parameters:
            path (str): Path to save the model (default: self.model_dir/best_model.keras).
        """
        self._check_model()
        path = path or os.path.join(self.model_dir, "final_model.keras")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: Optional[str] = None):
        """
        Load a model from disk.

        Parameters:
            path (str): Path to load the model from (default: self.model_dir/best_model.keras).
        """
        path = path or os.path.join(self.model_dir, "best_model.keras")
        self.model = tf.keras.models.load_model(path)

    def summary(self):
        """Print the model summary."""
        self._check_model()
        self.model.summary()

    def _check_model(self):
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")

# --- Example usage ---
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate synthetic multiclass data
    centers = [[-5, 2], [-2, -2], [1, 2], [5, 2]]
    X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=1.0, random_state=20)

    clf = SoftmaxClassifier(input_dim=X.shape[1], num_classes=4, hidden_layers=[32, 16], random_seed=42)
    clf.summary()

    # Train model
    clf.fit(X, y, epochs=30, validation_split=0.15, patience=3, checkpoint=True, verbose=1)

    # Evaluate
    metrics = clf.evaluate(X, y)
    print("Test Metrics:", metrics)

    # Predict probabilities and labels
    proba = clf.predict_proba(X[:5])
    print("Predicted probabilities (first 5):\n", proba)
    print("Predicted labels (first 5):", clf.predict(X[:5]))

    # Save and Load
    clf.save()  # Save final model
    clf.load()  # Reload best checkpoint
    print("Model loaded and ready.")
