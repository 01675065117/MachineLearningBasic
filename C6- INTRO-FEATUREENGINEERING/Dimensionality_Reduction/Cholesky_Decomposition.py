# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:43:05 2021

@author: Admin
"""
'''
Cholesky Decomposition
The Cholesky decomposition is for square symmetric matrices where all eigenvalues are greater than zero, so-called positive definite matrices.

A = L . L^T
Where A is the matrix being decomposed, L is the lower triangular matrix and L^T is the transpose of L.

The decompose can also be written as the product of the upper triangular matrix, for example:
A = U^T . U
Where U is the upper triangular matrix.

Decomposition MA TRẬN ĐỐI XỨNG
Chỉ hiệu quả với ma trận đối xứng
'''


# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define a 3x3 matrix
A = array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
print(A)
# Cholesky decomposition
L = cholesky(A)
print(L)
# reconstruct
B = L.dot(L.T)
print(B)























