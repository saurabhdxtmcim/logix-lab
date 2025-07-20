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

# ------------------------------------------------------------------------------
#  Evaluating a Learning Algorithm (Polynomial Regression)
#
# Let's say you have created a machine learning model and you find it *fits*
# your training data very well. You're done? Not quite.
# The goal of creating the model was to be able to predict values for *new* examples.
#
# How can you test your model's performance on new data before deploying it?
# The answer has two parts:
#
# 1. Split your original data set into "Training" and "Test" sets.
#    - Use the training data to fit the parameters of the model
#    - Use the test data to evaluate the model on *new* data
#
# 2. Develop an error function to evaluate your model.
#
# Example: Polynomial Regression
#   f_w,b(x) = w1 * x1 + w2 * x^2 + ... + wn * x^n + b
#
# The diagram typically shows training data vs new data (in different colors)
# and how the learned curve fits the training data but must be tested against unseen data.
# ------------------------------------------------------------------------------

#splitting your data

X, y, x_ideal, y_ideal = gen_data(m=18, seed=2, scale=0.7)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# split the data using sklearn routine

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# plot the data
plt_train_test(X_train, y_train, X_test, y_test, X, y, x_ideal, y_ideal, degree=2)

# ------------------------------------------------------------------------------
#  Error Calculation for Model Evaluation - Linear Regression
#
# When *evaluating* a linear regression model, you average the squared error
# difference between the predicted values and the actual target values.
#
# The error function used for this evaluation is:
#
#       J_test(w, b) = (1 / (2 * m_test)) * Σ[i=0 to m_test-1] (f_w,b(x_test^(i)) - y_test^(i))^2
#
# Where:
#   - m_test: number of test samples
#   - f_w,b(x): the predicted value for input x using weights w and bias b
#   - y_test: the actual target value
#
# This is also referred to as the Mean Squared Error (MSE) on the test set.
# ------------------------------------------------------------------------------

def eval_mse(y, yhat):
    """
    Calculates the mean squared error (MSE) between target and predicted values.

    Args:
        y (ndarray): Target values, shape (m,) or (m, 1).
        yhat (ndarray): Predicted values, shape (m,) or (m, 1).

    Returns:
        float: Mean squared error as a scalar.
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        err += (yhat[i] - y[i])**2
    return err / (2 * m)

# test the function

yhat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 2.0, 3.0, 4.0, 6.0])

print(f"MSE: {eval_mse(y, yhat)}")

# compare performance on training dataset
# create a model in sklearn, train on the training data

degree = 10
lmodel = lin_model(degree=degree)
lmodel.fit(X_train, y_train)

# predict on training data, find training error
yhat =  lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat=yhat)

print(f"Training error: {err_train}")

# predict on test data, find test error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat=yhat)
print(f"Test error: {err_test}")

# plot the data
# ------------------------------------------------------------------------------
# Plot predictions over the data range using the provided plt_train_test utility.
# ------------------------------------------------------------------------------

# Generate a range of x values covering the training and test data
x_min = min(np.min(X_train), np.min(X_test))
x_max = max(np.max(X_train), np.max(X_test))
x_plot = np.linspace(x_min, x_max, 200)

# Predict using the trained model over the generated x values
y_plot = lmodel.predict(x_plot)

# Plot using the assignment utility function
plt_train_test(
    X_train, y_train,
    X_test, y_test,
    x_plot, y_plot,
    x_ideal, y_ideal,
    degree
)

# ------------------------------------------------------------------------------
#  Data Splitting: Training, Cross-Validation, and Test Sets
#
# For robust model evaluation and tuning, it is common to split the data into three sets:
#   - Training set (60%): Used to fit model parameters (e.g., weights, bias).
#   - Cross-validation set (20%): Used to tune hyperparameters (e.g., polynomial degree, regularization).
#   - Test set (20%): Used to evaluate the final model's performance on unseen data.
#
# This can be achieved by calling train_test_split from sklearn twice:
#   1. First, split the data into training (60%) and temp (40%).
#   2. Then, split the temp set equally into cross-validation (20%) and test (20%).
# ------------------------------------------------------------------------------

# lets generate 3 datasets and use train_test_split from sklearn
# gennerate the data

X, y, x_ideal, y_ideal = gen_data(m=40, seed=5, scale=0.7)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# split the data using sklearn routine
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_cv shape: {X_cv.shape}, y_cv shape: {y_cv.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ------------------------------------------------------------------------------
# Bias and Variance: Diagnosing Model Performance
#
# To choose a good model complexity (e.g., polynomial degree), compare training and
# cross-validation errors:
#   - High bias (underfit): Both training and cross-validation errors are high and similar.
#   - High variance (overfit): Training error is low, but cross-validation error is much higher.
#   - Optimum: Cross-validation error is minimized at an intermediate model complexity.
#
# By evaluating performance across a range of model complexities, you can select the
# degree that best balances bias and variance.
# ------------------------------------------------------------------------------

# Plot training, cross-validation, and test data along with the ideal curve and model prediction
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x_ideal, y_ideal, label='ideal', color='black', lw=1)
ax.scatter(X_train, y_train, color='blue', label='train', s=20)
ax.scatter(X_cv, y_cv, color='green', label='cv', s=20)
ax.scatter(X_test, y_test, color='red', label='test', s=20)
ax.plot(x_plot, y_plot, color='magenta', lw=1, label='model')
ax.set_title("Training, CV, Test Error", fontsize=12)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Finding the Optimal Degree: Overfitting and Underfitting
#
# By increasing the degree of a polynomial model, you can fit more complex curves.
# However, a higher degree can also lead to overfitting, where the model fits the
# training data too closely and performs poorly on new data.
#
# To distinguish between overfitting and underfitting:
#   - Train the model repeatedly, increasing the polynomial degree each iteration.
#   - Evaluate performance on training, cross-validation, and test sets.
#   - Use scikit-learn's linear regression for speed and simplicity.
#
# This approach helps identify the degree that best balances model complexity and
# generalization, allowing you to select the optimal degree for your data.
# ------------------------------------------------------------------------------

max_degree = 9
err_train = np.zeros(max_degree)    
err_cv = np.zeros(max_degree)      
x = np.linspace(0,int(X.max()),100)  
y_pred = np.zeros((100,max_degree))  #columns are lines to plot

for degree in range(max_degree):
    lmodel = lin_model(degree+1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:,degree] = lmodel.predict(x)
    
optimal_degree = np.argmin(err_cv)+1
print(f"Optimal degree: {optimal_degree}")

# plot the data
plt.close("all")
plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, err_train, err_cv, optimal_degree, max_degree)

# ------------------------------------------------------------------------------
# Interpreting Model Complexity and Error Curves
#
# By separating data into groups (training and cross-validation), we can determine
# if a model is underfitting or overfitting as we vary the polynomial degree.
#
# - On the left plot: Solid lines show model predictions. A low-degree polynomial
#   (e.g., degree 1) fits only the general trend, while a high-degree polynomial
#   fits nearly every data point.
#
# - On the right plot:
#     - Training error (blue) decreases as model complexity increases.
#     - Cross-validation error (green) decreases at first, then increases as the
#       model starts to overfit the training data (fails to generalize).
#
# Note: The specific curves may not be perfectly smooth due to the data, but the
# general trend is what matters for diagnosing underfitting and overfitting.
# ------------------------------------------------------------------------------

#------------------------------------------------------------------------------

# Tuning Regularization
# ------------------------------------------------------------------------------
# Tuning Regularization to Reduce Overfitting
#
# Regularization is used to reduce overfitting by adding a penalty term to the loss function.
# Similar to tuning the polynomial degree, you can tune the regularization parameter (lambda, λ)
# to find the best balance between bias and variance.
#
# The same methodology applies: train the model with different values of λ and evaluate
# performance on training and cross-validation sets.
#
# Example: Start with a high-degree polynomial and vary the regularization parameter to
# observe its effect on model performance.
# ------------------------------------------------------------------------------

lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)    
err_cv = np.zeros(num_steps)       
x = np.linspace(0,int(X.max()),100) 
y_pred = np.zeros((100,num_steps))  #columns are lines to plot

for i in range(num_steps):
    lambda_= lambda_range[i]
    lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)
    y_pred[:,i] = lmodel.predict(x)
    
optimal_reg_idx = np.argmin(err_cv) 

# plot the data
plt.close("all")
plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)

# ------------------------------------------------------------------------------
# Interpreting Regularization Tuning Plots
#
# The plots above show how increasing regularization (lambda) affects model performance:
#   - As regularization increases, the model transitions from high variance (overfitting)
#     to high bias (underfitting).
#   - The right plot displays the training and cross-validation errors as lambda varies.
#   - The vertical line indicates the optimal value of lambda, where cross-validation
#     error is minimized.
#
# In this example, the polynomial degree was set to 10.
# ------------------------------------------------------------------------------ 

# Increasing training size
# When a model is overfitting(high variance), increasing the training size can help.

X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)

# ------------------------------------------------------------------------------
# Effect of Increasing Training Size on Model Performance
#
# The plots above show that when a model has high variance (overfitting), adding more
# training examples improves performance and generalization. As the number of examples
# increases, the training and cross-validation errors converge to similar values.
#
# - On the left plot: The final curve with the highest value of m is a smooth curve
#   centered in the data.
# - On the right plot: As the number of examples increases, the performance of the
#   training and cross-validation sets converge.
#
# Note: The curves may not be perfectly smooth due to the data, but the trend is clear:
# more data improves generalization.
#
# Important: Adding more examples does NOT improve performance when the model has
# high bias (underfitting).
# ------------------------------------------------------------------------------

