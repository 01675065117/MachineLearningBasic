# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:01:15 2022

@author: Admin
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Iris dataset from sklearn
X, y = datasets.load_iris(return_X_y=True)

kf = KFold(n_splits=5)
for train, test in kf.split(X, y):
    print("%s %s" % (train,test))















