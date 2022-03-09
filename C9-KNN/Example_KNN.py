# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:55:34 2022

@author: Admin
"""

from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('labels: ', np.unique(iris_y))

X_train, X_test, y_train, y_test = train_test_split(iris_X,iris_y, test_size=130)
print('Train size: ', X_train.shape[0], ', Test size: ',X_test.shape[0])

model = neighbors.KNeighborsClassifier(n_neighbors = 7, p =2, weights = 'distance') # p = 2 is the l2 norm, p=1 is the l1 norm
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))





