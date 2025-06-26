# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

# Utilize scikit-learn to implement linear regression using Gradient Descent
# ------------------------------------------------------------------------------
# This code implements linear regression using Scikit-Learn's SGDRegressor, 
# which is a stochastic gradient descent-based linear regression model.
# It includes feature scaling and visualizes the predictions against actual values.
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import os

# ------------------------------------------------------------------------------
def load_housing_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data/housing_data.txt')
    data = np.loadtxt(data_path, delimiter=',', skiprows = 1)
    x = data[:, :4]  # Features: size, bedrooms, floor, age: selects all rows (:) and first four columns (:4), x becomes an array of shape (m, 4).
    y = data[:, 4] # Target: price: selects all rows and only the 5th column (index 4), y becomes a 1D array of shape (m,)
    return x, y
# ------------------------------------------------------------------------------

x_train, y_train = load_housing_data()
x_features = ['size(sqft)','bedrooms','floors','age']
# ------------------------------------------------------------------------------
# Feature Scaling: Standardization (Z-score normalization)
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)

print(f"Peak to Peak range by column in Raw        X:{np.ptp(x_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(x_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(x_norm)
# make a prediction using w,b. 
y_pred = np.dot(x_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

# ------------------------------------------------------------------------------
# Plotting the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('SGDRegressor: Predicted vs Actual Housing Prices')
plt.legend()
plt.grid(True)
plt.show()