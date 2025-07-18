# This model evaluation step is inspired by the bias-variance diagnostic approach:
# - High Bias (Underfitting): Both training error (J_train) and cross-validation error (J_cv) are high.
#   → The model is too simple to capture the underlying trend (e.g., linear fit to quadratic data).
# - High Variance (Overfitting): J_train is low, but J_cv is high.
#   → The model captures noise in the training data and performs poorly on unseen data.
# - Just Right: Both J_train and J_cv are low and close.
#   → The model generalizes well.

# Remedies:
# To fix high bias:
#   - Add more features or polynomial terms
#   - Reduce regularization
# To fix high variance:
#   - Increase regularization
#   - Use fewer features
#   - Add more training data

#--------------------------------
# Before diagnosing a model for high bias or variance, it's important to estimate a reasonable baseline level of performance.
# You can set this baseline using:
#   - Human-level performance (if available)
#   - Performance of competing algorithms
#   - A rough estimate based on experience

# Real-world datasets are often noisy, making 0% error unrealistic.
# For example, a training error of 10% and CV error of 15% might look like high bias,
# but if human-level performance is also around 10%, then the model is actually doing well.
# In this case, the issue is not high bias but high variance (CV error being too high).

# Always compare training/CV errors with a realistic performance floor to interpret bias/variance correctly.


from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

data = np.loadtxt('./data/c2w3_lab2_data1.csv', delimiter=',')

# Split the data into features and target variable
# Assuming the first column is the feature and the second column is the target variable
x = data[:, :-1]
y = data[:, -1]

# get 60% of the dataset for training set, put the remaining 40% in temporary variables for cross-validation and test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.4, random_state = 80)

#Split the 40% subset  above into two, one half for cross validation and the other half for the test set
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 80)

# delete the temporary variables
del X_temp, y_temp

print(f"X_train shape: {X_train.shape}")
print(f"X_cv shape: {X_cv.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_cv shape: {y_cv.shape}")
print(f"y_test shape: {y_test.shape}")

print(f" first five rows of X_train: {X_train[:5]} ")
print(f" first five rows of X_cv: {X_cv[:5]} ")
print(f" first five rows of X_test: {X_test[:5]} ")
print(f" first five rows of y_train: {y_train[:5]} ")
print(f" first five rows of y_cv: {y_cv[:5]}")
print(f" first five rows of y_test: {y_test[:5]}")

model = LinearRegression()

def plot_model_fit(model, X_train, y_train, X_cv, y_cv, X_test, y_test, degree=None, baseline=None):
    """
    Fits the given model to the training data, predicts on train/cv/test,
    and plots the data and model fit.
    
    Parameters:
    - model: The model to fit
    - X_train, y_train: Training data
    - X_cv, y_cv: Cross-validation data  
    - X_test, y_test: Test data
    - degree: Optional polynomial degree (default: None)
    - baseline: Optional baseline value for plotting (default: None)
    """

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the training, cross-validation, and test sets
y_train_pred = model.predict(X_train)
y_cv_pred = model.predict(X_cv)
y_test_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, color='blue', label='Train Data')
plt.scatter(X_cv[:, 0], y_cv, color='green', label='CV Data')
plt.scatter(X_test[:, 0], y_test, color='red', label='Test Data')

# Plot the model's predictions as a line over the full x range
x_all = np.concatenate([X_train[:, 0], X_cv[:, 0], X_test[:, 0]])
x_line = np.linspace(x_all.min(), x_all.max(), 100).reshape(-1, 1)

# For prediction, pad with zeros if model expects 2 features
if model.coef_.shape[0] > 1:
    x_line_full = np.zeros((100, model.coef_.shape[0]))
    x_line_full[:, 0] = x_line[:, 0]
else:
    x_line_full = x_line

y_line = model.predict(x_line_full)

plt.plot(x_line, y_line, color='black', linewidth=2, label='Model Prediction')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Model Fit and Data Splits')
plt.legend()
plt.grid(True)
plt.show()

plot_model_fit(model, X_train, y_train, X_cv, y_cv, X_test, y_test)

# now you see more plynomial features you add mode will fit the  training data well, but the cv and test data will not be fit well

def train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=None):
    
    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    degrees = range(1,max_degree+1)

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for degree in degrees:

        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model.fit(X_train_mapped_scaled, y_train )
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    # Plot the results
    plt.plot(degrees, train_mses, marker='o', c='r', label='training MSEs'); 
    plt.plot(degrees, cv_mses, marker='o', c='b', label='CV MSEs'); 
    plt.plot(degrees, np.repeat(baseline, len(degrees)), linestyle='--', label='baseline')
    plt.title("degree of polynomial vs. train and CV MSEs")
    plt.xticks(degrees)
    plt.xlabel("degree"); 
    plt.ylabel("MSE"); 
    plt.legend()
    plt.show()

# Train and plot polynomial regression models. Bias is defined lower.
train_plot_poly(model, X_train, y_train, X_cv, y_cv, max_degree=10, baseline=250)

# adding additional features
data = np.loadtxt('./data/c2w3_lab2_data2.csv', delimiter=',')

# Split the data into features and target variable
# Assuming the first column is the feature and the second column is the target variable
x = data[:, :-1]
y = data[:, -1]

# get 60% of the dataset for training set, put the remaining 40% in temporary variables for cross-validation and test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.4, random_state = 80)

#Split the 40% subset  above into two, one half for cross validation and the other half for the test set
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 80)


print(f"the shape of the training set (input) is: {X_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {X_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

# Preview the first 5 rows
print(f"first 5 rows of the training inputs (2 features):\n {X_train[:5]}\n")

# Now lets see what this does too the training process as we force the model to fit the training data well


model = LinearRegression()
plot_model_fit(model, X_train, y_train, X_cv, y_cv, X_test, y_test, degree=10, baseline=250)

# Try decreasing the regularization parameter
def train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 1, baseline=None):
    
    train_mses = []
    cv_mses = []
    models = []
    scalers = []

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for reg_param in reg_params:

        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model = Ridge(alpha=reg_param)
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    # Plot the results
    reg_params = [str(x) for x in reg_params]
    plt.plot(reg_params, train_mses, marker='o', c='r', label='training MSEs'); 
    plt.plot(reg_params, cv_mses, marker='o', c='b', label='CV MSEs'); 
    plt.plot(reg_params, np.repeat(baseline, len(reg_params)), linestyle='--', label='baseline')
    plt.title("lambda vs. train and CV MSEs")
    plt.xlabel("lambda"); 
    plt.ylabel("MSE"); 
    plt.legend()
    plt.show()

# Try decreasing the regularization parameter
# Define lambdas to plot
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
train_plot_reg_params(reg_params, X_train, y_train, X_cv, y_cv, degree=10, baseline=250)

# The resulting plot shows an initial λ of 10 and as you can see, the training error is worse than the baseline at that point.
# This implies that it is placing a huge penalty on the w parameters and this prevents the model from learning more complex patterns in your data.
# As you decrease λ, the model loosens this restriction and the training error is able to approach the baseline performance.

# Fixing High Variance
# You will now look at some things to try when your model has overfit the training set.
# The main objective is to have a model that generalizes well to new examples,
# so you want to minimize the cross validation error.

# Try increasing the regularization parameter
# In contrast to the last exercise above, setting a very small value of the regularization parameter
# will keep the model low bias but might not do much to improve the variance.
# As shown below, you can improve your cross validation error by increasing the value of λ.

# Define lambdas to plot
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# Define degree of polynomial and train for each value of lambda
train_plot_reg_params(reg_params, X_train, y_train, X_cv, y_cv, degree= 4, baseline=250)

# Try smaller sets of features
# You've already seen in the last lab that having too many polynomial terms can result in overfitting.
# You can reduce the number of such terms and see where you get the best balance of training and cross validation error.
# Another scenario where reducing the number of features would be helpful is when you have irrelevant features in your data.
# For example, patient IDs that hospitals give will not help in diagnosing a tumor so you should make sure to remove it from your training data.
#
# To illustrate how removing features can improve performance, you will do polynomial regression for 2 datasets:
# the same data you used above (2 features) and another with a random ID column (3 features).
# You can preview these using the cell below. Notice that 2 columns are identical and a 3rd one is added to include random numbers.

data = np.loadtxt('./data/c2w3_lab2_data2.csv', delimiter=',')

# Split the data into features and target variable
# Assuming the first column is the feature and the second column is the target variable
x = data[:, :-1]
y = data[:, -1]

# get 60% of the dataset for training set, put the remaining 40% in temporary variables for cross-validation and test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.4, random_state = 80)

#Split the 40% subset  above into two, one half for cross validation and the other half for the test set
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 80)

# Preview the first 5 rows
print(f"first 5 rows of the training set with 2 features:\n {X_train[:5]}\n")

# Prepare dataset with randomID feature
data = np.loadtxt('./data/c2w3_lab2_data3.csv', delimiter=',')

# Split the data into features and target variable
# Assuming the first column is the feature and the second column is the target variable
x = data[:, :-1]
y = data[:, -1]

# get 60% of the dataset for training set, put the remaining 40% in temporary variables for cross-validation and test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.4, random_state = 80)

#Split the 40% subset  above into two, one half for cross validation and the other half for the test set
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 80)

# Preview the first 5 rows
print(f"first 5 rows of the training set with 3 features (1st column is a random ID):\n {X_train[:5]}\n")

model = LinearRegression()

plot_model_fit(model, X_train, y_train, X_cv, y_cv, X_test, y_test, degree=10, baseline=250)


data = np.loadtxt('./data/c2w3_lab2_data4.csv', delimiter=',')

def train_plot_learning_curve(model, x_train, y_train, x_cv, y_cv, degree= 1, baseline=None):
    
    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    num_samples_train_and_cv = []
    percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for percent in percents:
        
        num_samples_train = round(len(x_train) * (percent/100.0))
        num_samples_cv = round(len(x_cv) * (percent/100.0))
        num_samples_train_and_cv.append(num_samples_train + num_samples_cv)
        
        x_train_sub = x_train[:num_samples_train]
        y_train_sub = y_train[:num_samples_train]
        x_cv_sub = x_cv[:num_samples_cv]
        y_cv_sub = y_cv[:num_samples_cv]

        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train_sub)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model.fit(X_train_mapped_scaled, y_train_sub)
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train_sub, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv_sub)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv_sub, yhat) / 2
        cv_mses.append(cv_mse)

    # Plot the results
    plt.plot(num_samples_train_and_cv, train_mses, marker='o', c='r', label='training MSEs'); 
    plt.plot(num_samples_train_and_cv, cv_mses, marker='o', c='b', label='CV MSEs'); 
    plt.plot(num_samples_train_and_cv, np.repeat(baseline, len(percents)), linestyle='--', label='baseline')
    plt.title("number of examples vs. train and CV MSEs")
    plt.xlabel("total number of training and cv examples"); 
    plt.ylabel("MSE"); 
    plt.legend()
    plt.show()
# Split the data into features and target variable
# Assuming the first column is the feature and the second column is the target variable
x = data[:, :-1]
y = data[:, -1]

# get 60% of the dataset for training set, put the remaining 40% in temporary variables for cross-validation and test
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.4, random_state = 80)

#Split the 40% subset  above into two, one half for cross validation and the other half for the test set
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 80)

# Preview the first 5 rows
print(f"first 5 rows of the training set with 3 features (1st column is a random ID):\n {X_train[:5]}\n")

model = LinearRegression()

train_plot_learning_curve(model, X_train, y_train, X_cv, y_cv, degree=10, baseline=250)