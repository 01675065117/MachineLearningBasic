# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:04:33 2022

@author: Admin
"""

from pandas import read_csv
from sklearn.linear_model import Ridge
# load the dataset
# The dataset involves predicting the house price given details of the house’s suburb in the American city of Boston.
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge(alpha=1.0)
#the lambda term can be configured via the “alpha” argument when defining the class. The default value is 1.0 or a full penalty.
# fit model
model.fit(X, y)
# define new data
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted: %.3f' % yhat)