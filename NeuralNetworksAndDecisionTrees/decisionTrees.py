# ------------------------------------------------------------------------------
# Decision Trees: Information Gain and Entropy
#
# In decision trees, we decide whether to split a node by looking at the 
# information gain that the split would provide.
#
# Information Gain = H(p^node) - (w^left * H(p^left) + w^right * H(p^right))
#
# Where H is the entropy, defined as:
# H(p1) = -p1 * log2(p1) - (1 - p1) * log2(1 - p1)
#
# Key properties of entropy H(p):
#   - Maximum value when p = 0.5 (maximum uncertainty)
#   - Minimum value when p = 0 or p = 1 (totally predictable)
#   - Shows the degree of predictability of an event
#
# Note: All logarithms are in base 2.
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ------------------------------------------------------------------------------
# Plot the entropy function
# ------------------------------------------------------------------------------
def plot_entropy():
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        else:
            return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    p_array = np.linspace(0,1,201)
    h_array = [entropy(p) for p in p_array]
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set_title('p x H(p)')
    ax.set_xlabel('p')
    ax.set_ylabel('H(p)')
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    h_plot = ax.plot(p_array,h_array)
    scatter = ax.scatter(0,0,color = 'red', zorder = 100, s = 70)
    slider = Slider(axfreq, 'p', 0, 1, valinit = 0, valstep = 0.05)

    def update(val):
        x = val
        y = entropy(x)
        scatter.set_offsets((x,y))

    slider.on_changed(update)
    return slider

# ------------------------------------------------------------------------------
# Plot the entropy function
# ------------------------------------------------------------------------------
_ = plot_entropy()
plt.show()

# ------------------------------------------------------------------------------
# Dataset: Animal Classification with One-Hot Encoding
#
# The dataset contains animals with the following features:
#   - Ear Shape: Pointy or Floppy
#   - Face Shape: Round or Not Round  
#   - Whiskers: Present or Absent
#   - Cat: 1 (is a cat) or 0 (not a cat)
#
# One-hot encoding for categorical features:
#   - Ear Shape: Pointy = 1, Floppy = 0
#   - Face Shape: Round = 1, Not Round = 0
#   - Whiskers: Present = 1, Absent = 0
#
# Data structure:
#   - X_train: Each example contains 3 features (Ear Shape, Face Shape, Whiskers)
#   - y_train: Binary classification (1 if cat, 0 otherwise)
# ------------------------------------------------------------------------------

X_train = np.array([[1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

# for instance the first example
print(f"First example: {X_train[0]}")
print(f"First example: {y_train[0]}")

# ------------------------------------------------------------------------------
# Plot the dataset
# ------------------------------------------------------------------------------
def plot_dataset(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c = y, cmap = 'bwr')
    ax.set_xlabel('Ear Shape')
    ax.set_ylabel('Face Shape')
    plt.show()

plot_dataset(X_train, y_train)

# ------------------------------------------------------------------------------
# Decision Tree Node Splitting Process
#
# For each node, we compute the information gain for each feature, then split the node
# on the feature with the highest information gain. This is done by comparing the entropy
# of the current node with the weighted entropy of the two resulting child nodes.
#
# Root Node Analysis:
# - The root node contains every animal in our dataset
# - p1^node = proportion of positive class (cats) in the root node
# - For our dataset: p1^node = 5/10 = 0.5
#
# The algorithm will evaluate splitting on each feature (Ear Shape, Face Shape, Whiskers)
# and choose the split that maximizes information gain.
# ------------------------------------------------------------------------------

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)

print(f"Entropy of root node: {entropy(0.5)}")

#lets compute the information gain if we split the node for each of the features
def split_indices(X, index_feature):
    """
        Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
        that feature = 1 and the right node those that have the feature = 0 
        index feature = 0 => ear shape
        index feature = 1 => face shape
        index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []

    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

print(split_indices(X_train, 0))

# ------------------------------------------------------------------------------
# Weighted Entropy Calculation for Split Nodes
#
# To compute the weighted entropy in split nodes, we need to find:
#   - w^left and w^right: the proportion of animals in each node
#   - p^left and p^right: the proportion of cats in each split
#
# Important distinction:
#   - w values represent the fraction of total examples in each split
#   - p values represent the fraction of positive class (cats) within each split
#
# Example: Splitting on Ear Shape (feature index 0)
# Left node (animals 0,3,4,5,7): w^left = 5/10 = 0.5, p^left = 4/5 = 0.8
# Right node (animals 1,2,6): w^right = 5/10 = 0.5, p^right = 1/5 = 0.2
# ------------------------------------------------------------------------------

def weighted_entropy(X, y, left_indices, right_indices):
    """
        Given a dataset, the target vector, and the indices of the two split nodes, return the weighted entropy of the two split nodes
    """
    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    p_left = np.mean(y[left_indices])
    p_right = np.mean(y[right_indices])
    return w_left * entropy(p_left) + w_right * entropy(p_right)

print(weighted_entropy(X_train, y_train, split_indices(X_train, 0)[0], split_indices(X_train, 0)[1]))

left_indices, right_indices = split_indices(X_train, 0)
print(weighted_entropy(X_train, y_train, left_indices, right_indices))

# ------------------------------------------------------------------------------
# Information Gain Calculation
#
# For each feature, we calculate the information gain by comparing the entropy of the current node
# with the weighted entropy of the two resulting child nodes.
# To compute the Information Gain we must subtract it from the entropy in the node we chose to split (in this case, the root node).
# ------------------------------------------------------------------------------

def information_gain(X, y, left_indices, right_indices):
    """
        Here X has the elements in the node and y is theirs respective classes
        left_indices and right_indices are the indices of the two split nodes
    """
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy

print(f"Information gain for the split: {information_gain(X_train, y_train, left_indices, right_indices)}")

# ------------------------------------------------------------------------------

# lets split the information gain if we split the root nnode for each of the features

for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Information gain for the split on {feature_name}: {i_gain}")
    print(f"Weighted entropy for the split on {feature_name}: {weighted_entropy(X_train, y_train, left_indices, right_indices)}")
    print(f"Entropy (left child) on {feature_name}: {entropy(np.mean(y_train[left_indices]))}")
    print(f"Entropy (right child) on {feature_name}: {entropy(np.mean(y_train[right_indices]))}")


# ------------------------------------------------------------------------------
# Function to find the best feature to split on
# ------------------------------------------------------------------------------

def get_best_split(X, y, node_indices):
    """
    Find the feature that gives the highest information gain for splitting
    """
    best_feature = 0
    best_gain = 0
    
    for feature in range(X.shape[1]):  # Loop through all features
        left_indices, right_indices = split_indices(X[node_indices], feature)
        # Convert to actual indices
        left_actual = [node_indices[i] for i in left_indices]
        right_actual = [node_indices[i] for i in right_indices]
        
        gain = information_gain(X, y, left_actual, right_actual)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    
    return best_feature

# ------------------------------------------------------------------------------
# not build the tree recursively, we will use the information gain to split the node

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth, tree):
    """
        Build a decision tree recursively
    """
    if current_depth == max_depth:
        print(f"Reached leaf: {branch_name} with samples: {node_indices}")
        return
    best_split_feature = get_best_split(X, y, node_indices)
    X_node = X[node_indices]
    left_sub, right_sub = split_indices(X_node, best_split_feature)
    left_indices = [node_indices[i] for i in left_sub]
    right_indices = [node_indices[i] for i in right_sub]

    tree.append((branch_name, best_split_feature, left_indices, right_indices))
    build_tree_recursive(X, y, left_indices, f"{branch_name}_left", max_depth, current_depth + 1, tree)
    build_tree_recursive(X, y, right_indices, f"{branch_name}_right", max_depth, current_depth + 1, tree)

tree = []
"""
Visualize the tree structure in a simple way by printing the splits and their indices.
"""

build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=5, current_depth=0, tree=tree)

for node in tree:
    branch_name, split_feature, left_indices, right_indices = node
    print(f"Node: {branch_name}")
    print(f"  Split on feature: {split_feature}")
    print(f"  Left indices: {left_indices}")
    print(f"  Right indices: {right_indices}")
    print("-" * 40)

