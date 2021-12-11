# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:19:45 2021

@author: Admin
"""

'''
LU Matrix Decomposition
The LU decomposition is for square matrices and decomposes a matrix into L and U components.
A = L.U or A = LU
Trong đó A là ma trận vuông mà chúng ta muốn phân rã, L là ma trận tam giác dưới và U là ma trận tam giác trên.
A = P.L.U
Các hàng của ma trận mẹ được sắp xếp lại thứ tự để đơn giản hóa quá trình phân rã và ma trận P bổ sung chỉ định một cách để hoán vị kết quả hoặc trả kết quả về thứ tự ban đầu.
'''
# LU decomposition
from numpy import array
from scipy.linalg import lu
import cv2
# define a square matrix
#A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
img = cv2.imread('D:\Tai-Lieu-Hoc\TNCKH\Graduation_Thesis\MLB\MachineLearningBasic\Media\image.JPG',0)
A = array(img)
print(A)
# LU decomposition
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
B = P.dot(L).dot(U)
print(B)




