# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:48:40 2022

@author: Admin
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

model1 = Ridge(alpha=1.0)
# fit model
model1.fit(X, y)

print("scikit-learn: w_1 = ", model1.coef_[0], "w_0 = ", model1.intercept_)

yhat = model1.predict([[160]])

# summarize prediction
print('Predicted: %.3f' % yhat)


