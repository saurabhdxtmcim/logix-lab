# SD: this file is doing a lot, would be dividing it into multiple files, this is here just to learn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
np.set_printoptions(precision=3, suppress=True)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')
# Split the data into features and target variable
# Assuming the first column is the feature and the second column is the target variable
X = data[:, 0]
y = data[:, 1]

# COnvert 1-D arrays into 2D because commands later will require it
x = np.expand_dims(X, axis = 1)
y = np.expand_dims(y, axis = 1)




# PLOT THE DATA
plt.scatter(x, y, color='blue', label='Data points')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Data Scatter Plot')
plt.legend()
plt.show()

"""
Split the dataset into training, cross validation, and test sets

In previous labs, you might have used the entire dataset to train your models.
In practice, however, it is best to hold out a portion of your data to measure
how well your model generalizes to new examples. This will let you know if
the model has overfit to your training set.

As mentioned in the lecture, it is common to split your data into three parts:

    • training set - used to train the model
    • cross validation set (also called validation, development, or dev set) -
      used to evaluate the different model configurations you are choosing from.
      For example, you can use this to make a decision on what polynomial
      features to add to your dataset.
    • test set - used to give a fair estimate of your chosen model's performance
      against new examples. This should not be used to make decisions while you
      are still developing the models.

Scikit-learn provides a train_test_split function to split your data into the
parts mentioned above. In the code below, you will split the entire dataset
into 60% training, 20% cross validation, and 20% test.
"""
# Split the dataset into training, cross-validation, and test sets
# get 60% training, 20% cross-validation, and 20% test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
# Now split the remaining 40% into 20% cross-validation and 20% test
# This will give us 60% training, 20% cross-validation, and 20
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#delete X_temp, y_temp  # Free up memory
del X_temp, y_temp

print(f"Training set size: {X_train.shape[0]}")
print(f"Cross-validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Cross-validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# Plot training, cross-validation (dev), and test data
plt.scatter(X_train, y_train, color='blue', label='Training set')
plt.scatter(X_val, y_val, color='green', label='Cross-validation set')
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Train, Cross-validation, and Test Data')
plt.legend()
plt.show()

"""
Fit a linear model

Now that you have split the data, one of the first things you can try is to fit a linear model.
You will do that in the next sections below.

Feature scaling

In the previous course of this specialization, you saw that it is usually a good idea to perform
feature scaling to help your model converge faster. This is especially true if your input features
have widely different ranges of values. Later in this lab, you will be adding polynomial terms,
so your input features will indeed have different ranges. For example, x runs from around 1600
to 3600, while x^2 will run from 2.56 million to 12.96 million.

You will only use x for this first model, but it's good to practice feature scaling now so you can
apply it later. For that, you will use the StandardScaler class from scikit-learn. This computes
the z-score of your inputs. As a refresher, the z-score is given by the equation:

    z = (x - μ) / σ

where μ is the mean of the feature values and σ is the standard deviation. The code below shows how
to prepare the training set using the said class. You can plot the results again to inspect if it
still follows the same pattern as before. The new graph should have a reduced range of values for x.
"""

scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_scaled = scaler_linear.fit_transform(X_train)

print(f"Mean of training set: {scaler_linear.mean_.squeeze():<.2f}")
print(f"Standard deviation of training set: {scaler_linear.scale_.squeeze():<.2f}")

#plot the scaled training set
plt.scatter(X_train_scaled, y_train, color='blue', label='Scaled Training set')
plt.xlabel('Scaled Feature')
plt.ylabel('Target')
plt.title('Scaled Training Data Scatter Plot')
plt.legend()
plt.show()

"""
Train the model

Next, you will create and train a regression model. For this lab, you will use scikit‑learn's LinearRegression
class to fit your data. Note that there are other linear regressors available in scikit‑learn (e.g., Ridge,
Lasso, ElasticNet) which you can explore later.
"""

linear_model = LinearRegression()
# Fit the model to the scaled training data
linear_model.fit(X_train_scaled, y_train)

"""
Evaluate the Model

To assess how well your regression model performs, measure the error on both training and validation sets.

1. Mean Squared Error (MSE) formula for training set:
       J_train(w, b) = (1 / (2 * m_train)) * Σᵢ (f_{w,b}(x_train⁽ⁱ⁾) - y_train⁽ⁱ⁾)²
   - m_train is the number of training examples.
   - We divide by 2*m to match the convention from the cost-function lectures.

2. scikit-learn's mean_squared_error():
   - By default, divides by m (not 2*m).
   - To match our formula, compute:
         mse = mean_squared_error(y_true, y_pred)
         J = mse / 2

3. Implementation notes:
   - You can use 'mean_squared_error()' for a quick calculation, then divide by 2.
   - We also include a pure Python for-loop version to verify equivalence.
   - Since the model was trained on scaled (z-score) inputs, be sure to feed in the scaled versions when computing errors.
"""

#feed the scaled training set to the model to get predictions
yhat = linear_model.predict(X_train_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# for-loop implementation
total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i  = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i                                              

mse = total_squared_error / (2*len(yhat))

print(f"training MSE (for-loop implementation): {mse.squeeze()}")

"""
Evaluate on Cross‐Validation Set

1. Compute MSE for cross‐validation:
       J_cv(w, b) = (1 / (2 * m_cv)) * Σᵢ (f_{w,b}(x_cv[i]) - y_cv[i])²
   - m_cv is the number of cross‐validation examples.

2. Feature scaling for CV set:
   - Use the same mean (μ) and standard deviation (σ) computed on the training set.
   - This ensures new inputs are transformed exactly as during training.

3. Intuition scenario:
   • Training feature x = 500  →  z = (500 - μ_train) / σ_train = 0.5  
   • Model maps z = 0.5  →  y = 300  
   • New input x = 500 must use μ_train and σ_train to get z = 0.5  
   • If you use different μ/σ, you get a different z and likely a wrong prediction.

4. Implementation snippet:
       X_cv_scaled = scaler.transform(X_cv)
       y_cv_pred    = model.predict(X_cv_scaled)
       mse_cv_half  = compute_mse_half(y_cv, y_cv_pred)  # uses 1/(2*m_cv) convention
"""

X_cv_scaled = scaler_linear.transform(X_val )  # Use the same scaler as for training

print(f"Mean of cross-validation set: {scaler_linear.mean_.squeeze():<.2f}")
print(f"Standard deviation of cross-validation set: {scaler_linear.scale_.squeeze():<.2f}")
# Get predictions for the cross-validation set
y_cv_pred = linear_model.predict(X_cv_scaled)
# Compute MSE for cross-validation set using scikit-learn's utility function
mse_cv = mean_squared_error(y_val, y_cv_pred) / 2
print(f"Cross-validation MSE (using sklearn function): {mse_cv}")

"""
Adding Polynomial Features

Based on your earlier plots, the target y increases more sharply at smaller x values
and flattens out as x grows. A straight line may not capture this curvature. Now that
you have the training and cross‑validation MSE for your linear model, you can try
adding polynomial terms to improve performance. The overall code stays mostly the
same, with a few extra preprocessing steps.

Create the Additional Features

Use scikit‑learn’s PolynomialFeatures class to generate new input features from your
training set. For example, with degree=2, it will add a feature equal to x² alongside
the original x, so your model can learn quadratic relationships.
"""
#Instantiate the class to make polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)    
# Fit the polynomial features to the training data and transform it
X_train_poly = poly_features.fit_transform(X_train_scaled)

print(f"Polynomial features shape: {X_train_poly.shape}")

#instantiate the class
scaler_poly = StandardScaler()
# Fit the scaler to the polynomial features and transform it
X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
# Preview the first 5 elements of the scaled training set.
print(X_train_poly_scaled[:5])

# Adding Polynomial Features
#
# Based on your earlier plots, the target y increases more sharply at smaller x values
# and flattens out as x grows. A straight line may not capture this curvature. Now that
# you have the training and cross‑validation MSE for your linear model, you can try
# adding polynomial terms to improve performance. The overall code stays mostly the
# same, with a few extra preprocessing steps.
#
# Create the Additional Features
#
# Use scikit‑learn’s PolynomialFeatures class to generate new input features from your
# training set. For example, with degree=2, it will add a feature equal to x² alongside
# the original x, so your model can learn quadratic relationships.

#Initialize the class

model = LinearRegression()
# Fit the model to the scaled polynomial training data
model.fit(X_train_poly_scaled, y_train)

#Compute the training MSE
y_train_poly_pred = model.predict(X_train_poly_scaled)
mse_train_poly = mean_squared_error(y_train, y_train_poly_pred) / 2
print(f"Training MSE (polynomial features): {mse_train_poly}")
# Compute the cross-validation MSE
X_val_poly = poly_features.transform(X_val)
X_val_poly_scaled = scaler_poly.transform(X_val_poly)
y_val_poly_pred = model.predict(X_val_poly_scaled)
mse_cv_poly = mean_squared_error(y_val, y_val_poly_pred) / 2
print(f"Cross-validation MSE (polynomial features): {mse_cv_poly}")

# Model Comparison: Polynomial Degrees 1–10
#
# • You’ll notice that adding a 2nd‑order term (x²) significantly lowered both
#   training and cross‑validation MSEs.
# • To find the best fit, we can extend this idea to polynomials of higher degree.
#
# Polynomial model definitions:
#   1)  f(x) = w₁·x + b
#   2)  f(x) = w₁·x + w₂·x² + b
#   3)  f(x) = w₁·x + w₂·x² + w₃·x³ + b
#   … 
#  10)  f(x) = w₁·x + w₂·x² + … + w₁₀·x¹⁰ + b
#
# Loop strategy (degree = 1 to 10):
#   1. Generate polynomial features: 
#        poly = PolynomialFeatures(degree=d, include_bias=False)
#        X_train_d = poly.fit_transform(X_train)
#        X_cv_d    = poly.transform(X_cv)
#   2. Scale features:
#        scaler = StandardScaler().fit(X_train_d)
#        X_train_d_scaled = scaler.transform(X_train_d)
#        X_cv_d_scaled    = scaler.transform(X_cv_d)
#   3. Fit model:
#        model = LinearRegression().fit(X_train_d_scaled, y_train)
#   4. Compute MSE (1/(2·m) convention) on train and CV sets
#   5. Store MSEs for plotting
#
# Finally, plot training vs. cross‑validation MSE as a function of polynomial degree
# to compare performance and select the optimal degree.

#initialize lists to save the errors, models and feature transform
train_mse = []
cv_mse = []
models = []
polys = []
scalers = []

#Loop through polynomial degrees from 1 to 10
for degree in range(1, 11):
    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_d = poly.fit_transform(X_train)
    X_cv_d = poly.transform(X_val)

    # Scale features
    scaler = StandardScaler().fit(X_train_d)
    X_train_d_scaled = scaler.transform(X_train_d)
    X_cv_d_scaled = scaler.transform(X_cv_d)

    # Fit model
    model = LinearRegression().fit(X_train_d_scaled, y_train)

    # Compute MSE (1/(2·m) convention) on train and CV sets
    train_mse.append(mean_squared_error(y_train, model.predict(X_train_d_scaled)) / 2)
    cv_mse.append(mean_squared_error(y_val, model.predict(X_cv_d_scaled)) / 2)

    # Store models and feature transforms
    models.append(model)
    polys.append(poly)
    scalers.append(scaler)
# Plot training vs. cross-validation MSE as a function of polynomial degree
plt.plot(range(1, 11), train_mse, label='Training MSE', marker='o')
plt.plot(range(1, 11), cv_mse, label='Cross-validation MSE', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training vs. Cross-validation MSE by Polynomial Degree')
plt.legend()
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# Choosing the Best Model
#
# When selecting a model, aim for one that performs well on both the training and
# cross‑validation sets—i.e., it learns the underlying patterns without overfitting.
#
# In this lab you’ll observe:
#   • A sharp drop in cross‑validation error from degree=1 to degree=2
#   • A relatively flat validation error curve from degree=2 through degree=5
#   • An increase in validation error beyond degree=5 (signs of overfitting)
#
# Use these insights to pick the model (polynomial degree) with the lowest cv_mse
# as the best fit for your application.

degree_best = np.argmin(cv_mse) + 1  # +1 because degrees start from 1
print(f"The best polynomial degree is: {degree_best}")
print(f"Lowest CV MSE is found in the model with degree={degree_best} and is {cv_mse[degree_best - 1]}")

# Publish Generalization Error on Test Set
#
# • Transform the test features just like training and CV:
#       X_test_scaled = scaler.transform(X_test)
#
# • Make predictions on the scaled test data:
#       y_test_pred = model.predict(X_test_scaled)
#
# • Compute the test MSE using the 1/(2*m) convention:
#       mse_test = compute_mse_half(y_test, y_test_pred)
#
# This gives your final estimate of how well the model generalizes to unseen data.

x_test_mapped = polys[degree_best - 1].transform(X_test)
x_test_scaled = scalers[degree_best - 1].transform(x_test_mapped)

#compute the test mse
y_test_pred = models[degree_best - 1].predict(x_test_scaled)
mse_test = mean_squared_error(y_test, y_test_pred) / 2

print(f"Test MSE for the best model (degree={degree_best}): {mse_test}")
# Plot the predictions of the best model against the test data
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.scatter(X_test, y_test_pred, color='blue', label='Predictions')
plt.xlabel('Input Feature')
plt.ylabel('Target Variable')
plt.title('Test Data vs Predictions')
plt.legend()
plt.show()

# Neural Networks
# The same model selection process can also be used when choosing between different neural network architectures.
# In this section, you will create the models shown below and apply them to the same regression task as before.
#
# Architectures:
# 1. Input → Dense(25 units) → Dense(15 units) → Output(1 unit)
# 2. Input → Dense(20 units) → Dense(12 units) → Dense(12 units) → Dense(20 units) → Output(1 unit)
# 3. Input → Dense(32 units) → Dense(16 units) → Dense(8 units) → Dense(4 units) → Dense(12 units) → Output(1 unit)
#
# Prepare the Data
# You will use the same training, cross‑validation, and test sets you generated in the previous section.
# From earlier lectures in this course, you may know that neural networks can learn non‑linear relationships,
# so you can opt to skip adding polynomial features.
# The code for polynomial feature expansion is still included below in case you want to try it later
# and see what effect it has on your results.
# The default `degree` is set to 1 to indicate that it will just use `x_train`, `x_cv`, and `x_test` as is
# (i.e., without any additional polynomial features).

# add polynomial features
degree = 1
poly = PolynomialFeatures(degree=degree, include_bias=False)
x_train_mapped = poly.fit_transform(X_train)
x_cv_mapped = poly.transform(X_val)
x_test_mapped = poly.transform(X_test)


# scaling the input features to help  gradient descent converge faster. 
# Again, notice that you are using the mean and standard deviation computed from the training set by just using transform() in the cross validation and 
# test sets instead of fit_transform()

scaler  = StandardScaler()
x_train_mapped_scaled = scaler.fit_transform(x_train_mapped)
x_cv_mapped_scaled = scaler.transform(x_cv_mapped)
x_test_mapped_scaled = scaler.transform(x_test_mapped)

# Build the models
#
# • Use the same model definition for each architecture.
# • Use the same training, cross‑validation, and test sets.
# • Use the same polynomial feature expansion.

# initiate the lists that will contain the errors for each model
nm_train_mses = []
nm_cv_mses = []

# Build the models
def build_models():
    tf.random.set_seed(1234)
    
    model1 = tf.keras.Sequential([
        #tf.keras.layers.Input(shape=(1,)),     Letting keras infer the input shape
        tf.keras.layers.Dense(units=25, activation='relu'),
        tf.keras.layers.Dense(units=15, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ], name='model1')
    model2 = tf.keras.Sequential([
        #tf.keras.layers.Input(shape=(1,)),     Letting keras infer the input shape
        tf.keras.layers.Dense(units=20, activation='relu'),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=20, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ], name='model2')
    model3 = tf.keras.Sequential([
        #tf.keras.layers.Input(shape=(1,)),     Letting keras infer the input shape
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=8, activation='relu'),
        tf.keras.layers.Dense(units=4, activation='relu'),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ], name='model3')

    return [model1, model2, model3]

nm_models = build_models()

# Loop over the models
for model in nm_models:
    # SETUP the loss and optimizer
    model.compile(
        loss = 'mse',
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    )
    print(f"Model {model.name} is ready")
    # Train the model
    model.fit(
        x = x_train_mapped_scaled,
        y = y_train,
        epochs = 100,
        verbose = 0
    )

    print("--------------------------------Done--------------------------------")

    yhat  =  model.predict(x_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    nm_train_mses.append(train_mse)

    # Record the cross-validation MSE
    yhat = model.predict(x_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_val, yhat) / 2
    nm_cv_mses.append(cv_mse)

#Print the training and cross-validation MSEs for each model
for i in range(len(nm_models)):
    print(f"Model {nm_models[i].name} has a training MSE of {nm_train_mses[i]} and a cross-validation MSE of {nm_cv_mses[i]}")

# select the model with the lowest CV MSE
model_num = 3

#Compute the test MSE
yhat = nm_models[model_num - 1].predict(x_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Selected model: {nm_models[model_num - 1].name} with a test MSE of {test_mse}")
print(f"Trainign MSE: {nm_train_mses[model_num - 1]}")
print(f"Cross-validation MSE: {nm_cv_mses[model_num - 1]}")
print(f"Test MSE: {test_mse}")

# Plot the training and cross-validation MSEs for each model
model_names = ['Model 1\n(25→15→1)', 'Model 2\n(20→12→12→20→1)', 'Model 3\n(32→16→8→4→12→1)']
plt.plot(range(1, 4), nm_train_mses, label='Training MSE', marker='o')
plt.plot(range(1, 4), nm_cv_mses, label='Cross-validation MSE', marker='o')
plt.xlabel('Neural Network Model')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training vs. Cross-validation MSE by Neural Network Architecture')
plt.legend()
plt.xticks(range(1, 4), model_names)
plt.grid()
plt.show()

# practice model evaluation and selection on a classification task. 
# The process will be similar, with the main difference being the computation of the errors. 
# You will see that in the following sections.

data = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')

# Split the data into features and labels
x_bc = data[:, :-1]
y_bc = data[:, -1]

# Convert y into a 2D array, because commands later will require it and x is already 2D
y_bc = np.expand_dims(y_bc, axis=1)

print(f"x_bc shape: {x_bc.shape}")
print(f"y_bc shape: {y_bc.shape}")

# Plot the data (2D feature space with color-coded labels)
# For classification, plot feature1 vs feature2, colored by class
plt.scatter(x_bc[:, 0], x_bc[:, 1], c=y_bc.ravel(), cmap='viridis', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2') 
plt.title('Classification Data: Feature 1 vs Feature 2')
plt.colorbar(label='Class')
plt.show()

# split and prepare the dataset 

from sklearn.model_selection import train_test_split

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x_bc_temp, y_bc_train, y_bc_temp = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

# Split the 40% temporary set into 20% validation and 20% test
x_bc_val, x_bc_test, y_bc_val, y_bc_test = train_test_split(x_bc_temp, y_bc_temp, test_size=0.50, random_state=1)

# delete the temporary variables
del x_bc_temp, y_bc_temp

# print the shapes of the datasets
print(f"x_bc_train shape: {x_bc_train.shape}")
print(f"x_bc_val shape: {x_bc_val.shape}")
print(f"x_bc_test shape: {x_bc_test.shape}")

# Scale the features
#Initialize the class scaler
scaler = StandardScaler()

# Fit the scaler on the training features and transform all three datasets
x_bc_train_scaled = scaler.fit_transform(x_bc_train)
x_bc_val_scaled = scaler.transform(x_bc_val)
x_bc_test_scaled = scaler.transform(x_bc_test)

"""
Evaluating the error for classification models

In the previous sections on regression models, you used the mean squared error to measure 
how well your model is doing. For classification, you can get a similar metric by getting 
the fraction of the data that the model has misclassified. For example, if your model made 
wrong predictions for 2 samples out of 5, then you will report an error of 40% or 0.4. 
The code below demonstrates this using a for-loop and also with Numpy's mean() function.
"""
# Sample model output
probabilities = np.array([0.2, 0.7, 0.1, 0.9, 0.3])

#Apply a threshold to the model output, If greater than 0.5 set to 1 else 0
predictions = np.where(probabilities > 0.5, 1, 0)
print(f"Predictions: {predictions}")

# ground truth labels
ground_truth = np.array([0, 1, 0, 1, 0])

# Initialize the error counter
error_count = 0

# get number of predictions
num_predictions = len(predictions)

for i in range(num_predictions):
    if predictions[i] != ground_truth[i]:
        error_count += 1
print(f"Error count: {error_count}")

# Coompute the fraction of the data that the model is misclassified
fraction_error = error_count / num_predictions

print(f"probabilities: {probabilities} \n predictions: {predictions} \n ground truth: {ground_truth}")
print(f"Fraction of the data that the model is misclassified: {fraction_error}")
print(f"fraction of misclassified data with np.mean: {np.mean(predictions != ground_truth)}")

"""
Build and train the model

You will use the same neural network architectures in the previous section so you can call the 
build_models() function again to create new instances of these models.

You will follow the recommended approach mentioned last week where you use a linear activation 
for the output layer (instead of sigmoid) then set from_logits=True when declaring the loss 
function of the model. You will use the binary_crossentropy_loss because this is a binary 
classification problem.

After training, you will use a sigmoid function to convert the model outputs into probabilities. 
From there, you can set a threshold and get the fraction of misclassified examples from the 
training and cross validation sets.
"""
#Initialize the lists that will contain the errors for each model
nn_train_error = []
nn_cv_error = []

#build the models
models_bc = build_models()

#loop over the models
for model in models_bc:
    #setup the loss and optimizer
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    )

    print(f"Model {model.name} is ready")
    #train the model
    model.fit(
        x = x_bc_train_scaled,
        y = y_bc_train,
        epochs = 100,
        verbose = 0
    )
    print("--------------------------------Done--------------------------------")
    
    #get the predictions
    yhat = model.predict(x_bc_train_scaled)
    #convert the predictions to probabilities
    yhat = tf.math.sigmoid(yhat) 
    yhat = np.where(yhat > 0.5, 1, 0)
    train_error = np.mean(yhat != y_bc_train)
    nn_train_error.append(train_error)
    #record the cross-validation error
    yhat = model.predict(x_bc_val_scaled)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat > 0.5, 1, 0)
    cv_error = np.mean(yhat != y_bc_val)
    nn_cv_error.append(cv_error)

# Print the results
for i in range(len(models_bc)):
    print(f"Model {models_bc[i].name} has a training error of {nn_train_error[i]} and a cross-validation error of {nn_cv_error[i]}")

# Select the model with the lowest error

model_num = 3

# Compute the test error
yhat = models_bc[model_num - 1].predict(x_bc_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat > 0.5, 1, 0)
test_error = np.mean(yhat != y_bc_test)

print(f"Selected model: {models_bc[model_num - 1].name} with a test error of {test_error}")
print(f"Training error: {nn_train_error[model_num - 1]}")
print(f"Cross-validation error: {nn_cv_error[model_num - 1]}")
print(f"Test error: {test_error}")
print(f"Test accuracy: {1 - test_error}")

import matplotlib.pyplot as plt

# Plot training and cross-validation errors for each model
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(models_bc) + 1), nn_train_error, marker='o', label='Training Error')
plt.plot(range(1, len(models_bc) + 1), nn_cv_error, marker='s', label='Cross-Validation Error')
plt.xlabel('Model Number')
plt.ylabel('Error')
plt.title('Training and Cross-Validation Error for Each Model')
plt.xticks(range(1, len(models_bc) + 1))
plt.legend()
plt.grid(True)
plt.show()




