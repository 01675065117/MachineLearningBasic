# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:15:36 2022

@author: Admin
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

regres = linear_model.LinearRegression()
regres.fit(X,y)

print("scikit-learn: w_1 = ", regres.coef_[0], "w_0 = ", regres.intercept_)



