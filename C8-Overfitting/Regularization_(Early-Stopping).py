# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:29:41 2022

@author: Admin
"""

from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model

#------Load and split data----------------
def loadData(n_samples=100, noise=0.2, random_state=1):
    # This dataset is called the “moons” dataset because of the shape of the observations in each class when plotted.
    X, y = make_moons(n_samples, noise, random_state)
    
    return X, y

def split(X, y, n_train = 30):
    #Split data
    X_train, X_test = X[:n_train, :],  X[n_train:, :]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test

def create_model(X_train, X_test, y_train, y_test):
    
    #define model
    model = Sequential()
    model.add(Dense(500, input_dim = 2, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    #fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4000, verbose=0, callbacks=[es])
    
    #saved_model = load_model('best_model.h5')
    
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    return history, model

def plot(history):
    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


X, y = loadData()
X_train, X_test, y_train, y_test = split(X, y)
history, model = create_model(X_train, X_test, y_train, y_test)
plot(history)















