# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------
# Description: This module contains a function to compute the sigmoid of a given input.
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))