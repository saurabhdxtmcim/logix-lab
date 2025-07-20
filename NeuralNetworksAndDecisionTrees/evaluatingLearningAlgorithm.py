import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# borrowed helper classes
from testHelper.assignmentUtils import *
from testHelper.publicTestHelper import *

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.keras.backend.set_floatx("float64")

tf.autograph.set_verbosity(0)

# Generate the datset

X, y, centers, classes, std = gen_blobs()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=1)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# ------------------------------------------------------------------------------
# Cluster Data, Ambiguity, and Model Evaluation
#
# The data consists of six clusters, each identified by color. Both training points
# (dots) and cross-validation points (triangles) are shown. Ambiguous points are those
# that fall in locations where either cluster might consider them members.
#
# - What would you expect a neural network model to do with ambiguous points?
# - What would be an example of overfitting? Underfitting?
#
# On the right, an 'ideal' model is shown, where boundaries are drawn at equal distances
# between cluster centers. This model, knowing the true source of the data, would still
# misclassify about 8% of the total dataset due to ambiguity.
# ------------------------------------------------------------------------------
plt_train_eq_dist(X_train, y_train, classes, X_cv, y_cv, centers, std)

# ------------------------------------------------------------------------------
# Evaluating Categorical Models: Classification Error
#
# The evaluation function for categorical models is the fraction of incorrect predictions:
#
#     J_cv = (1/m) * sum_{i=0}^{m-1} [1 if y_hat^(i) != y^(i) else 0]
#
# where:
#   - m is the number of examples
#   - y_hat^(i) is the predicted label for example i
#   - y^(i) is the true label for example i
#
# This metric simply counts the proportion of predictions that do not match the true labels.
# ------------------------------------------------------------------------------

def eval_cat_err(y, yhat):
    """
    Calculate the classification error on a data set
    args:
        y : ndarray shape(m) or (m, 1) target value of each example
        yhat : ndarray shape(m) or (m, 1) predicted value of each example
    return:
        err : scalar
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:
            incorrect += 1
        cerr = incorrect / m
    return cerr

y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )

# BEGIN UNIT TEST  
test_eval_cat_err(eval_cat_err)
# END UNIT TEST
# lets build 2 models, a complex one and a simple one

# Complex model

# ------------------------------------------------------------------------------
# Three-Layer Neural Network Model Specification
#
# Compose a three-layer model as follows:
#   - Dense layer with 120 units, relu activation
#   - Dense layer with 40 units, relu activation
#   - Dense layer with 6 units, linear activation (not softmax)
#
# Compile the model using:
#   - Loss: SparseCategoricalCrossentropy (with from_logits=True)
#   - Optimizer: Adam with learning rate 0.01
# ------------------------------------------------------------------------------

tf.random.set_seed(1234)
model = Sequential([
    Dense(120, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(40, activation='relu'),
    Dense(6, activation='linear')
], name="complex_model")

model.compile(optimizer=Adam(learning_rate=0.01), loss=SparseCategoricalCrossentropy(from_logits=True))

# BEGIN UNIT TEST
model.fit(
    X_train, y_train,
    epochs=100
)
# END UNIT TEST

# BEGIN UNIT TEST
model.summary()
# END UNIT TEST

model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(), axis=1)
plt_nn(model_predict, X_train, y_train, classes, X_cv, y_cv, suptitle="Complex Model")

# ------------------------------------------------------------------------------
# Two-Layer Neural Network Model Specification (Simple Model)
#
# Compose a two-layer model as follows:
#   - Dense layer with 6 units, relu activation
#   - Dense layer with 6 units, linear activation (not softmax)
#
# Compile the model using:
#   - Loss: SparseCategoricalCrossentropy (with from_logits=True)
#   - Optimizer: Adam with learning rate 0.01
# ------------------------------------------------------------------------------

tf.random.set_seed(1234)
model_s = Sequential([
    Dense(6, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(6, activation='linear')
], name="simple_model")

model_s.compile(optimizer=Adam(learning_rate=0.01), loss=SparseCategoricalCrossentropy(from_logits=True))
model_s.build(input_shape=(None, X_train.shape[1]))

# BEGIN UNIT TEST
model_s.fit(
    X_train, y_train,
    epochs=100
)
# END UNIT TEST

# BEGIN UNIT TEST
model_s.summary()
# END UNIT TEST

model_s_predict = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(), axis=1)
plt_nn(model_s_predict, X_train, y_train, classes, X_cv, y_cv, suptitle="Simple Model")

# ------------------------------------------------------------------------------
# Three-Layer Neural Network Model with Regularization
#
# Reconstruct the complex model, this time including L2 regularization:
#   - Dense layer with 120 units, relu activation, kernel_regularizer=tf.keras.regularizers.l2(0.1)
#   - Dense layer with 40 units, relu activation, kernel_regularizer=tf.keras.regularizers.l2(0.1)
#   - Dense layer with 6 units, linear activation (not softmax)
#
# Compile the model using:
#   - Loss: SparseCategoricalCrossentropy (with from_logits=True)
#   - Optimizer: Adam with learning rate 0.01
# ------------------------------------------------------------------------------

tf.random.set_seed(1234)
model_r = Sequential([
    Dense(120, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dense(6, activation='linear')
], name="regularized_model")

model_r.compile(optimizer=Adam(learning_rate=0.01), loss=SparseCategoricalCrossentropy(from_logits=True))

model_r.fit(
    X_train, y_train,
    epochs=100  
)

# BEGIN UNIT TEST
model_r.summary()
# END UNIT TEST

model_r_predict = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(), axis=1)
plt_nn(model_r_predict, X_train, y_train, classes, X_cv, y_cv, suptitle="Regularized Model")

training_cerr = eval_cat_err(y_train, model_r_predict(X_train))
cv_cerr = eval_cat_err(y_cv, model_r_predict(X_cv))
print(f"training error {training_cerr:0.3f}, cross-validation error {cv_cerr:0.3f}")

# ------------------------------------------------------------------------------
# Iterating to Find the Optimal Regularization Value
#
# As in linear regression, you can try many regularization values to find the best one.
# This process may take several minutes to run, depending on the number of values tested.
# If you have time, you can run the code and check the results.
# ------------------------------------------------------------------------------

tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)

for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")


plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)

plt_compare(X_test,y_test, classes, model_s_predict, model_r_predict, centers)