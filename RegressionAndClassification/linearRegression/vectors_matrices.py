# ------------------------------------------------------------------------------
# Copyright (c) 2025 Saurabh Dixit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# ------------------------------------------------------------------------------

import numpy as np
import time

#NumPy routines, which allocate memory and fill arrays with values
a = np.zeros(4) 
print("NumPy array a:", a)
a = np.zeros((4, ))
print("NumPy array a with shape (4,):", a)
a = np.random.random_sample(4 )
print("NumPy array a with random values:", a)
a = np.random.random_sample((4, ))
print("NumPy array a with random values and shape (4,):", a)

# data creation routines do not take a shape tuple
# NumPy routines, which allocate memory and fill arrays with values but do not accept shape as input parameter
a = np.arange(4.)  # Create an array with values from 0 to 3
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# Operations on Vectors

#Vector indexing operation on 1-D vector
a = np.arange(10)  # Create an array with values from 0 to 3
print(f"np.arange(10): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#accessing elements in a vector
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#Vector slicing operation on 1-D vector
a = np.arange(10)  # Create an array with values from 0 to 9
print(f"np.arange(10): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# Slicing the first 5 elements
b = a[:5]  # Slicing the first 5 elements
print(f"Slicing the first 5 elements: b = {b}, b shape = {b.shape}, b data type = {b.dtype}")
# Slicing the last 5 elements
b = a[-5:]  # Slicing the last 5 elements
print(f"Slicing the last 5 elements: b = {b}, b shape = {b.shape}, b data type = {b.dtype}")
# Slicing elements from index 2 to 5 (exclusive)
b = a[2:5]  # Slicing elements from index 2 to 5 (exclusive)    
print(f"Slicing elements from index 2 to 5: b = {b}, b shape = {b.shape}, b data type = {b.dtype}")
# Slicing elements from index 2 to the end
b = a[2:]  # Slicing elements from index 2 to the end
print(f"Slicing elements from index 2 to the end: b = {b}, b shape = {b.shape}, b data type = {b.dtype}")
#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)
# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)
# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)
# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)
# access all elements
c = a[:];         print("a[:]     = ", c)

#Single vector operations
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

#Vector addition
a = np.arange(10)  # Create an array with values from 0 to 9
b = np.arange(10, 20)  # Create another array with values from 10 to 19
print(f"Vector a: {a}, Vector b: {b}")
c = a + b  # Element-wise addition
print(f"Element-wise addition: c = {c}, c shape = {c.shape}, c data type = {c.dtype}")
#Vector subtraction
c = b - a  # Element-wise subtraction
print(f"Element-wise subtraction: c = {c}, c shape = {c.shape}, c data type = {c.dtype}")
#Vector multiplication
c = a * b  # Element-wise multiplication
print(f"Element-wise multiplication: c = {c}, c shape = {c.shape}, c data type = {c.dtype}")
#Vector division
c = b / (a + 1)  # Element-wise division, adding 1 to avoid division by zero
print(f"Element-wise division: c = {c}, c shape = {c.shape}, c data type = {c.dtype}")
#Vector exponentiation
c = a ** 2  # Element-wise exponentiation
print(f"Element-wise exponentiation: c = {c}, c shape = {c.shape}, c data type = {c.dtype}")

#Vector dot product
# Mathematical expression:
#       n-1
#   x = Σ (a_i * b_i)
#      i=0
#
# This is the dot product of two vectors a and b.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b)  # Dot product
print(f"Dot product of a and b: {dot_product}")
#Vector cross product
cross_product = np.cross(a, b)  # Cross product
print(f"Cross product of a and b: {cross_product}")

# Matrices: Matrices, are two dimensional arrays

#Matrix creation: Matrices include a second index. The two indexes describe [row, column]. Access can either return an element or a row/column
# Create a 2D array (matrix) with shape (3, 4)
matrix_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"2D Matrix:\n{matrix_2d}\nShape: {matrix_2d.shape}, Data type: {matrix_2d.dtype}")
print(f"Accessing element at (1, 2): {matrix_2d[1, 2]}")  # Accessing an element
print(f"Accessing row 1: {matrix_2d[1]}")  # Accessing a row

# Slicing a matrix
# Slicing the first two rows and first three columns
matrix_slice = matrix_2d[:2, :3]  # Slicing the first two rows and first three columns
print(f"Sliced Matrix (first 2 rows, first 3 columns):\n{matrix_slice}\nShape: {matrix_slice.shape}, Data type: {matrix_slice.dtype}")
# Slicing the last row and all columns
matrix_slice = matrix_2d[-1, :]  # Slicing the last row and all columns
print(f"Sliced Matrix (last row, all columns):\n{matrix_slice}\nShape: {matrix_slice.shape}, Data type: {matrix_slice.dtype}")
# Slicing all rows and the first two columns
matrix_slice = matrix_2d[:, :2]  # Slicing all rows and the first two columns
print(f"Sliced Matrix (all rows, first 2 columns):\n{matrix_slice}\nShape: {matrix_slice.shape}, Data type: {matrix_slice.dtype}")
