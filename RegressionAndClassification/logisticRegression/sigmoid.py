# Author: Saurabh Dixit
# Date: 2025-06-25
# Description: This module contains a function to compute the sigmoid of a given input.
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))