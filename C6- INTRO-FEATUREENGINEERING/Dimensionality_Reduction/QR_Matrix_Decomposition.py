# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:30:37 2021

@author: Admin
"""

'''
QR Matrix Decomposition
The QR decomposition is for m x n matrices (not limited to square matrices) 
A = Q.R or A = QR
Where A is the matrix that we wish to decompose, Q a matrix with the size m x m, and R is an upper triangle matrix with the size m x n.


'''
# QR decomposition
from numpy import array
from numpy.linalg import qr
import cv2
# define a 3x2 matrix
#A = array([[1, 2], [3, 4], [5, 6]])
img = cv2.imread('D:\Tai-Lieu-Hoc\TNCKH\Graduation_Thesis\MLB\MachineLearningBasic\Media\image.JPG',0)
A = array(img)
print(A)
# QR decomposition
Q, R = qr(A, 'complete')
print(Q)
print(R)
# reconstruct
B = Q.dot(R)
print(B)









