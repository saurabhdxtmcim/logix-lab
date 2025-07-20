import numpy as np
import matplotlib.pyplot as plt


X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print(f"X_train shape: {X_train.shape}")
print(f"First few examples: {X_train[:5]}")
print(f"First few labels: {y_train[:5]}")

# ------------------------------------------------------------------------------
# Exercise: Implementing compute_entropy Function
#
# First, you'll write a helper function called compute_entropy that computes the 
# entropy (measure of impurity) at a node.
#
# The function takes in a numpy array (y) that indicates whether the examples in 
# that node are edible (1) or poisonous (0).
#
# Implementation steps:
#   1. Compute p1, which is the fraction of examples that are edible (i.e. have value = 1 in y)
#   2. The entropy is then calculated as: H(p1) = -p1*log2(p1) - (1-p1)*log2(1-p1)
#
# Special cases to handle:
#   - The log is calculated with base 2
#   - For implementation purposes, 0*log2(0) = 0. That is, if p_1 = 0 or p_1 = 1, set the entropy to 0
#   - Make sure to check that the data at a node is not empty (i.e. len(y) != 0). Return 0 if it is
# ------------------------------------------------------------------------------

def compute_entropy(y):
    """
    Compute the entropy of a node
    """
    entropy = 0
    if len(y) == 0:
        return 0
    y1 = 0

    for i in range(len(y)):
        if y[i] == 1:
            y1 = y1+1
    p1 = y1/len(y)

    if p1 == 0 or p1 == 1:
        return 0
    entropy = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
    return entropy

print(f"Entropy: {compute_entropy(y_train[:5])}")

# ------------------------------------------------------------------------------
# Exercise: Implementing split_dataset Function
#
# Next, you'll write a helper function called split_dataset that takes in the data 
# at a node and a feature to split on and splits it into left and right branches.
# Later in the lab, you'll implement code to calculate how good the split is.
#
# The function takes in:
#   - The training data
#   - The list of indices of data points at that node
#   - The feature to split on
#
# It splits the data and returns the subset of indices at the left and the right branch.
#
# For example, say we're starting at the root node (so node_indices = [0,1,2,3,4,5,6,7,8,9])
# and we chose to split on feature 0, which is whether or not the example has a brown cap.
#   - The output of the function is then:
#     left_indices = [0,1,2,3,4,7,9] (data points with brown cap)
#     right_indices = [5,6,8] (data points without a brown cap)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Exercise 2: split_dataset Function Implementation Details
#
# Please complete the split_dataset() function shown below:
#
# For each index in node_indices:
#   - If the value of X at that index for that feature is 1, add the index to left_indices
#   - If the value of X at that index for that feature is 0, add the index to right_indices
#
# This function will separate the data points based on the binary feature value,
# creating two branches for the decision tree split.
# ------------------------------------------------------------------------------

def split_dataset(X, node_indices, feature):
    """ splitthe data at given node into left and right branches
    """
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

left_indices, right_indices = split_dataset(X_train, [0,1,2,3,4,5,6,7,8,9], 0)
print(f"Left indices: {left_indices}")
print(f"Right indices: {right_indices}")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Exercise 3: Calculate Information Gain
#
# Next, you'll write a function called information_gain that takes in the training data, 
# the indices at a node and a feature to split on and returns the information gain from the split.
#
# Please complete the compute_information_gain() function shown below to compute:
#
# Information Gain = H(p1^node) - (w^left * H(p1^left) + w^right * H(p1^right))
#
# where:
#   - H(p1^node) is entropy at the node
#   - H(p1^left) and H(p1^right) are the entropies at the left and right branches resulting from the split
#   - w^left and w^right are the proportion of examples at the left and right branch, respectively
#
# Note:
#   - You can use the compute_entropy() function that you implemented above to calculate the entropy
#   - We've provided some starter code that uses the split_dataset() function you implemented above to split the dataset
#
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
# ------------------------------------------------------------------------------

def compute_information_gain(X, y, node_indices, feature):
    """
        Compute the information of splitting the node on a given feature

        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            feature (int):           Index of feature to split on

        Returns:
        cost (float):        Cost computed

    """
   
    # Split the dataset into left and right branches
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    information_gain = 0

    # compute the entropy of the node
    entropy_node = compute_entropy(y_node)

    # compute the entropy of the left and right branches
    entropy_left = compute_entropy(y_left)
    entropy_right = compute_entropy(y_right)

    # compute the information gain
    information_gain = entropy_node - (len(left_indices) / len(node_indices) * entropy_left + len(right_indices) / len(node_indices) * entropy_right)

    return information_gain

print(f"Information gain: {compute_information_gain(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], 0)}")

# ------------------------------------------------------------------------------
# Decision Tree Implementation Summary
#
# This file demonstrates the core components of a decision tree algorithm:
#
# 1. compute_entropy(y): Calculates the entropy (measure of impurity) at a node
#    - Returns 0 for pure nodes (all examples have the same label)
#    - Uses the formula: H(p1) = -p1*log2(p1) - (1-p1)*log2(1-p1)
#
# 2. split_dataset(X, node_indices, feature): Splits data based on a binary feature
#    - Left branch: examples where feature value = 1
#    - Right branch: examples where feature value = 0
#    - Returns the indices for each branch
#
# 3. compute_information_gain(X, y, node_indices, feature): Measures split quality
#    - Information Gain = H(node) - weighted_average(H(left), H(right))
#    - Higher information gain indicates a better split
#    - Used to select the best feature for splitting at each node
#
# These functions form the foundation for building decision trees that can
# classify data by recursively splitting on the most informative features.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Exercise 4: Find the best split
#
# Next, you'll write a function called get_best_split that takes in the training data, 
# the indices at a node and returns the feature that results in the largest information gain.
#
# Please complete the get_best_split() function shown below:

def get_best_split(X, y, node_indices):
    """
    Compute the best split criteria for a node
    """
    best_feature = 0
    best_info_gain = 0
    for feature in range(X.shape[1]):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature

print(f"Best feature: {get_best_split(X_train, y_train, [0,1,2,3,4,5,6,7,8,9])}")






