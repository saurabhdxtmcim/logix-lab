# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# # Problem Statement
# <img align="left" src="./images/C1_W1_L3_S1_trainingdata.png"    style=" width:380px; padding: 10px;  " /> 
# 
# As in the lecture, you will use the motivating example of housing price prediction.  
# This lab will use a simple data set with only two data points - a house with 1000 square feet(sqft) sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000. These two points will constitute our *data or training set*. In this lab, the units of size are 1000 sqft and the units of price are 1000s of dollars.
# 
# | Size (1000 sqft)     | Price (1000s of dollars) |
# | -------------------| ------------------------ |
# | 1.0               | 300                      |
# | 2.0               | 500                      |
# 
# You would like to fit a linear regression model (shown above as the blue straight line) through these two points, so you can then predict price for other houses - say, a house with 1200 sqft.

x_train = np.array([1.0, 2.0, 3, 5, 8, 10])  # Input variable (size in 1000 square feet)
y_train = np.array([300.0, 500.0, 600, 700, 800, 1000])  # Target (price in 1000s of dollars)

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# ### Number of training examples `m`
m = x_train.shape[0]  # Number of training examples
print(f"Number of training examples is: {m}")

#plotting the training data
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.title('Training Data for Housing Price Prediction')
plt.legend()
plt.grid()  
#plt.show()

# model function for linear regression: 𝑓𝑤,𝑏(𝑥(𝑖)) = 𝑤𝑥(𝑖) + 𝑏
#The formula above is how you can represent straight lines - different values of  𝑤 and  𝑏
# give you different straight lines on the plot.

def model(x, w, b):
    """
    Linear model function for linear regression.
    
    Parameters:
    The argument description (ndarray (m,)) describes a Numpy n-dimensional array of shape (m,). (scalar) w, b describes an argument without dimensions, just a magnitude.
    
    Returns:
    numpy array: Predicted values based on the linear model.
    """
    m = x.shape[0]  # Number of training examples
    if m == 0:
        raise ValueError("Input array x is empty.")
    if not isinstance(w, (int, float)):
        raise TypeError("Weight w must be a number.")
    if not isinstance(b, (int, float)):
        raise TypeError("Bias b must be a number.")
    if x.ndim != 1:
        raise ValueError("Input x must be a 1-dimensional array.")
    if np.any(np.isnan(x)):
        raise ValueError("Input x contains NaN values.")
    if np.any(np.isinf(x)):
        raise ValueError("Input x contains infinite values.")
    if np.any(x < 0):
        raise ValueError("Input x contains negative values, which is not valid for size in sqft.")
    if w < 0:
        raise ValueError("Weight w must be non-negative.")
    if b < 0:
        raise ValueError("Bias b must be non-negative.")
    # Calculate the predicted values using the linear model
    if m == 1:
        return np.array([w * x[0] + b])
    else:
        # For multiple training examples, apply the model to each element
        # w * x + b is same as np.dot(w, x) + b or doing it with the loop
        return w * x + b
    

w = 80.0  # Initial weight (slope of the line)
b = 250.0  # Initial bias (y-intercept of the line)
w = 68.22344464899386
b =315.6758549231874
# Calculate the predicted values using the model function
y_pred = model(x_train, w, b)

print(f"Predicted values (y_pred) = {y_pred}")

# Plotting the training data and the model prediction
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.plot(x_train, y_pred, color='red', label='Model Prediction')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s of dollars)')
plt.title('Training Data and Model Prediction')
plt.legend()
plt.grid()
plt.show()

# now if I want to predict the price of a house with 1200 sqft, I can use the model function
def predict_price(size_sqft, w, b):
    """
    Predict the price of a house given its size using the linear model.
    
    Parameters:
    size_sqft (float): Size of the house in 1000 sqft.
    w (float): Weight (slope) of the linear model.
    b (float): Bias (y-intercept) of the linear model.
    
    Returns:
    float: Predicted price in 1000s of dollars.
    """
    return model(np.array([size_sqft]), w, b)[0]

# Predicting the price of a house with 1200 sqft
size_sqft = 1.2  # Size of the house in 1000 sqft
predicted_price = predict_price(size_sqft, w, b)
print(f"Predicted price for a house with {size_sqft} sqft is: {predicted_price} (in 1000s of dollars)")