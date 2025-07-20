import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 55
# ------------------------------------------------------------------------------
# Heart Failure Prediction Dataset
#
# Dataset: Obtained from Kaggle - Heart Failure Prediction Dataset
#
# Context:
# Cardiovascular disease (CVDs) is the number one cause of death globally, taking
# an estimated 17.9 million lives each year (31% of all deaths worldwide). Heart
# failure is a common event caused by CVDs. People with cardiovascular disease or
# at high cardiovascular risk need early detection and management.
#
# This dataset contains 11 features that can be used to predict possible heart disease:
#   - Age: age of the patient [years]
#   - Sex: sex of the patient [M: Male, F: Female]
#   - ChestPainType: chest pain type [TA, ATA, NAP, ASY]
#   - RestingBP: resting blood pressure [mm Hg]
#   - Cholesterol: serum cholesterol [mm/dl]
#   - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
#   - RestingECG: resting electrocardiogram results [Normal, ST, LVH]
#   - MaxHR: maximum heart rate achieved [60-202]
#   - ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
#   - Oldpeak: oldpeak = ST [depression value]
#   - ST_Slope: slope of the peak exercise ST segment [Up, Flat, Down]
#   - HeartDisease: output class [1: heart disease, 0: Normal]
#
# Categorical variables (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
# must be one-hot encoded before training machine learning models.
# ------------------------------------------------------------------------------

# load the dataset using pandas
df = pd.read_csv('./data/heart_failure_clinical_records_dataset.csv')

# display the first 5 rows of the dataset
print(df.head())

# ------------------------------------------------------------------------------
# One-Hot Encoding with Pandas
#
# Categorical variables need to be converted to numerical format for machine learning.
# Pandas provides get_dummies() function for one-hot encoding:
#   - Creates binary columns for each category
#   - Automatically handles categorical variables
#   - Drops original categorical columns by default
#
# For this dataset, categorical variables include:
#   - Sex (M/F)
#   - ChestPainType (TA/ATA/NAP/ASY)
#   - RestingECG (Normal/ST/LVH)
#   - ExerciseAngina (Y/N)
#   - ST_Slope (Up/Flat/Down)
# ------------------------------------------------------------------------------
cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df = pd.get_dummies(df, columns=cat_variables, prefix=cat_variables)
print(df.head())

# ------------------------------------------------------------------------------
# Feature Selection and Target Definition
#
# Let's choose the variables that will be the input features of the model:
#   - Target variable: HeartDisease (what we want to predict)
#   - Input features: All other variables that can potentially be used to predict HeartDisease
#
# After one-hot encoding, all variables except HeartDisease will serve as features
# for training the machine learning models.
# ------------------------------------------------------------------------------
features = [col for col in df.columns if col not in ['HeartDisease']] # Removing the target variable
print("Number of input features after one-hot encoding:", len(features))

#help( train_test_split)
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], test_size=0.8, random_state=RANDOM_STATE)
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

print(f'train samples: {len(X_train)}, val samples: {len(X_val)}')
# ------------------------------------------------------------------------------
# Building the Models: Decision Tree
#
# In this section, we'll work with the Decision Tree using the Scikit-learn implementation.
# There are several hyperparameters in the Decision Tree object from Scikit-learn.
# We will use only some of them and will not perform feature selection or hyperparameter
# tuning in this lab (but you are encouraged to do so and compare the results).
#
# The hyperparameters we will use and investigate here are:
#   - min_samples_split: The minimum number of samples required to split an internal node.
#     * Choosing a higher min_samples_split can reduce the number of splits and may help reduce overfitting.
#   - max_depth: The maximum depth of the tree.
#     * Choosing a lower max_depth can reduce the number of splits and may help reduce overfitting.
# ------------------------------------------------------------------------------

min_sample_split_list = [2, 10, 30, 50, 100, 200, 300, 400, 500, 700, 1000] # If thhe number is an integer, then it is the actual quantity of samples,
max_depth_list = [2, 3, 4, 8, 16, 64, None] # None means no limit on the depth
accuracy_list_train = []
accuracy_list_val = []

for min_samples_split in min_sample_split_list:
   # you can fitt the model at the same time you define it, because the fit function returns the fitted estimator
   model = DecisionTreeClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE)
   model.fit(X_train, y_train)
   predictions_train = model.predict(X_train) # predict the class of the training set
   predictions_val = model.predict(X_val) # predict the class of the validation set
   accuracy_train = accuracy_score(y_train, predictions_train)
   accuracy_val = accuracy_score(y_val, predictions_val)
   accuracy_list_train.append(accuracy_train) # append the accuracy of the training set
   accuracy_list_val.append(accuracy_val) # append the accuracy of the validation set
   print(f"min_samples_split: {min_samples_split}, max_depth: {model.max_depth}, accuracy_train: {accuracy_train}, accuracy_val: {accuracy_val}")

plt.plot(min_sample_split_list, accuracy_list_train, label='Training Accuracy')
plt.plot(min_sample_split_list, accuracy_list_val, label='Validation Accuracy')
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy')
plt.title('Accuracy vs min_samples_split')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# Analyzing min_samples_split Effects on Overfitting
#
# Note how increasing the number of min_samples_split reduces overfitting:
#   - Increasing min_samples_split from 10 to 30, and from 30 to 50, even though
#     it does not improve the validation accuracy, it brings the training accuracy
#     closer to it, showing a reduction in overfitting.
#
# Let's do the same experiment with max_depth to observe similar effects on
# model generalization and overfitting reduction.
# ------------------------------------------------------------------------------

accuracy_list_train = []
accuracy_list_val = []

for max_depth in max_depth_list:
   model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
   model.fit(X_train, y_train)
   predictions_train = model.predict(X_train)
   predictions_val = model.predict(X_val)
   accuracy_train = accuracy_score(y_train, predictions_train)
   accuracy_val = accuracy_score(y_val, predictions_val)
   accuracy_list_train.append(accuracy_train)
   accuracy_list_val.append(accuracy_val)
   print(f"max_depth: {max_depth}, accuracy_train: {accuracy_train}, accuracy_val: {accuracy_val}")

plt.plot(max_depth_list, accuracy_list_train, label='Training Accuracy')
plt.plot(max_depth_list, accuracy_list_val, label='Validation Accuracy')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs max_depth')
plt.legend()
plt.show()  

# ------------------------------------------------------------------------------
# Analyzing max_depth Effects and Optimal Hyperparameter Selection
#
# We can see that in general, reducing max_depth can help to reduce overfitting:
#   - Reducing max_depth from 8 to 4 increases validation accuracy closer to training accuracy,
#     while significantly reducing training accuracy.
#   - The validation accuracy reaches the highest at tree_depth=4.
#   - When max_depth is smaller than 3, both training and validation accuracy decreases.
#     The tree cannot make enough splits to distinguish positives from negatives (underfitting).
#   - When max_depth is too high (>= 5), validation accuracy decreases while training accuracy
#     increases, indicating that the model is overfitting to the training set.
#
# So we can choose the best values for these two hyperparameters for our model to be:
#   - max_depth = 4
#   - min_samples_split = 50
# ------------------------------------------------------------------------------        

decision_tree_model = DecisionTreeClassifier(max_depth=4, min_samples_split=50, random_state=RANDOM_STATE)
decision_tree_model.fit(X_train, y_train)

print(f"Decision Tree Model: {decision_tree_model}")
print(f"Metric train: {accuracy_score(y_train, decision_tree_model.predict(X_train))}   ")
print(f"Metric val: {accuracy_score(y_val, decision_tree_model.predict(X_val))}")

# ------------------------------------------------------------------------------
# Random Forest Algorithm Details
#
# Now let's try the Random Forest algorithm also, using the Scikit-learn implementation.
#
# Key characteristics of Random Forest:
#   - All hyperparameters found in the decision tree model will also exist in this algorithm,
#     since a random forest is an ensemble of many Decision Trees.
#   - One additional hyperparameter for Random Forest is called n_estimators, which is
#     the number of Decision Trees that make up the Random Forest.
#
# Random Forest Training Process:
#   - We randomly choose a subset of the features AND randomly choose a subset of the
#     training examples to train each individual tree.
#   - Following the lectures, if n is the number of features, we will randomly select
#     √n of these features to train each individual tree.
#   - Note that you can modify this by setting the max_features parameter.
#
# Performance Optimization:
#   - You can speed up your training jobs with another parameter, n_jobs.
#   - Since the fitting of each tree is independent of each other, it is possible to
#     fit more than one tree in parallel.
#   - Setting n_jobs higher will increase how many CPU cores it will use. Note that
#     numbers very close to the maximum cores of your CPU may impact overall performance
#     and even lead to freezes.
#   - Changing this parameter does not impact the final result but can reduce training time.
#
# We will run the same script again, but with another parameter, n_estimators, where
# we will choose between 10, 50, and 100. The default is 100.
# ------------------------------------------------------------------------------

min_sample_split_list = [2, 10, 30, 50, 100, 200, 300, 400, 500, 700, 1000] # If thhe number is an integer, then it is the actual quantity of samples,
max_depth_list = [2, 3, 4, 8, 16, 64, None] # None means no limit on the depth
n_estimator_list = [10, 50, 100, 500]

accuracy_list_train = []
accuracy_list_val = []

for min_samples_split in min_sample_split_list:
    #now you can fit the model at the same time you define it, because the fit function returns the fitted estimator
    model = RandomForestClassifier(min_samples_split = min_samples_split, random_state = RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(y_train, predictions_train)
    accuracy_val = accuracy_score(y_val, predictions_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
    print(f"min_samples_split: {min_samples_split}, accuracy_train: {accuracy_train}, accuracy_val: {accuracy_val}")

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(min_sample_split_list)), labels=[str(x) for x in min_sample_split_list])  # min_sample_split_list converted to str for labels
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()

# ------------------------------------------------------------------------------
# Random Forest Overfitting Analysis
#
# Notice that, even though the validation accuracy reaches the same both at
# min_samples_split = 2 and min_samples_split = 10, in the latter the difference
# in training and validation set reduces, showing less overfitting.
#
# This demonstrates that Random Forest, as an ensemble method, is naturally more
# robust against overfitting compared to individual decision trees, but tuning
# hyperparameters can still further improve generalization.
# ------------------------------------------------------------------------------

accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(y_train, predictions_train)
    accuracy_val = accuracy_score(y_val, predictions_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
    print(f"max_depth: {max_depth}, accuracy_train: {accuracy_train}, accuracy_val: {accuracy_val}")

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()


accuracy_list_train = []
accuracy_list_val = []
"""
Iterates over a list of n_estimators values to train and evaluate RandomForestClassifier models.
Appends training and validation accuracy for each n_estimators value to accuracy_list_train and accuracy_list_val, respectively.
Prints the n_estimators value along with corresponding training and validation accuracy.
"""
n_estimators_list = [10, 50, 100, 200, 300]  # Example list; adjust as needed

for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(y_train, predictions_train)
    accuracy_val = accuracy_score(y_val, predictions_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)
    
plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()

# ------------------------------------------------------------------------------
# Final Random Forest Model with Optimal Parameters
#
# Let's then fit a random forest with the following parameters:
#   - max_depth: 16
#   - min_samples_split: 10
#   - n_estimators: 100
#
# These parameters represent a good balance between model complexity and generalization,
# based on our hyperparameter analysis above.
# ------------------------------------------------------------------------------

random_forest_model = RandomForestClassifier(max_depth=16, min_samples_split=10, n_estimators=100, random_state=RANDOM_STATE)
random_forest_model.fit(X_train, y_train)

print(f"Random Forest Model: {random_forest_model}")
print(f"Metric train: {accuracy_score(y_train, random_forest_model.predict(X_train))}")
print(f"Metric val: {accuracy_score(y_val, random_forest_model.predict(X_val))}")

# ------------------------------------------------------------------------------
# Hyperparameter Tuning Strategy: One-at-a-Time vs Grid Search
#
# Note that we are searching for the best value one hyperparameter while leaving 
# the other hyperparameters at their default values.
#
# Ideally, we would want to check every combination of values for every hyperparameter 
# that we are tuning:
#   - If we have 3 hyperparameters, and each hyperparameter has 4 values to try out, 
#     we should have a total of 4 x 4 x 4 = 64 combinations to try.
#   - When we only modify one hyperparameter while leaving the rest as their default 
#     value, we are trying 4 + 4 + 4 = 12 results.
#
# To try out all combinations, we can use a sklearn implementation called GridSearchCV. 
# GridSearchCV has a refit parameter that will automatically refit a model on the best 
# combination so we will not need to program it explicitly. 
# For more on GridSearchCV, please refer to its documentation.
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# XGBoost: Gradient Boosting Algorithm
#
# Next is the Gradient Boosting model, called XGBoost. The boosting methods train 
# several trees, but instead of them being uncorrelated to each other, now the 
# trees are fit one after the other in order to minimize the error.
#
# The model has the same parameters as a decision tree, plus the learning rate:
#   - The learning rate is the size of the step on the Gradient Descent method 
#     that the XGBoost uses internally to minimize the error on each train step.
#
# Key XGBoost Features:
#   - During fitting, it can take in an evaluation dataset of the form (X_val, y_val).
#   - On each iteration, it measures the cost (or evaluation metric) on the evaluation datasets.
#   - Once the cost (or metric) stops decreasing for a number of rounds (called early_stopping_rounds), 
#     the training will stop.
#   - More iterations lead to more estimators, and more estimators can result in overfitting.
#   - By stopping once the validation metric no longer improves, we can limit the number 
#     of estimators created, and reduce overfitting.
#
# First, let's define a subset of our training set (we should not use the test set here).
# ------------------------------------------------------------------------------

n = int(len(X_train) * 0.8) # 80% of the training set for training
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

# ------------------------------------------------------------------------------
# XGBoost Training with Early Stopping
#
# We can then set a large number of estimators, because we can stop if the cost 
# function stops decreasing.
#
# Note some of the .fit() parameters:
#   - eval_set = [(X_train_eval, y_train_eval)]: Here we must pass a list to the 
#     eval_set, because you can have several different tuples of eval sets.
#   - early_stopping_rounds: This parameter helps to stop the model training if its 
#     evaluation metric is no longer improving on the validation set. It's set to 10.
#
# Early Stopping Mechanism:
#   - The model keeps track of the round with the best performance (lowest evaluation metric). 
#     For example, let's say round 16 has the lowest evaluation metric so far.
#   - Each successive round's evaluation metric is compared to the best metric. If the 
#     model goes 10 rounds where none have a better metric than the best one, then 
#     the model stops training.
#   - The model is returned at its last state when training terminated, not its state 
#     during the best round. For example, if the model stops at round 26, but the best 
#     round was 16, the model's training state at round 26 is returned, not round 16.
#   - Note that this is different from returning the model's "best" state (from when 
#     the evaluation metric was the lowest).
# ------------------------------------------------------------------------------

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=1, early_stopping_rounds=10)
xgb_model.fit(X_train_fit, y_train_fit, eval_set=[(X_train_eval, y_train_eval)])

print(f"Best iteration: {xgb_model.best_iteration}")

# ------------------------------------------------------------------------------
# XGBoost Training with Early Stopping
#
# We can then set a large number of estimators, because we can stop if the cost function stops decreasing.
# ------------------------------------------------------------------------------

xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=1, early_stopping_rounds=10)
xgb_model.fit(X_train_fit, y_train_fit, eval_set=[(X_train_eval, y_train_eval)])

print(f"Best iteration: {xgb_model.best_iteration}")

# ------------------------------------------------------------------------------