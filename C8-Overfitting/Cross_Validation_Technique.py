# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:10:45 2022

@author: Admin
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Iris dataset from sklearn
X, y = datasets.load_iris(return_X_y=True)

'''
The following example demonstrates how to estimate the accuracy of a linear kernel support vector machine
on the iris dataset by splitting the data, fitting a model and computing the score 5 consecutive
times (with different splits each time):
'''
clf = svm.SVC(kernel='linear' , C=1, random_state=42 )
scores = cross_val_score(clf, X, y, cv = 5)


scores1 = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')

print("Score: %0.6f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("f1_macro score: %0.6f accuracy with a standard deviation of %0.2f" % (scores1.mean(), scores1.std()))





