# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:45:54 2022

@author: Khoa
"""

import cv2
import numpy as np
import os
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import dump, load

# Load the Classifier, class names, scaler, number of clusters and vocabulary
# From stored pickle file (generated during trining)
clf, classes_names, stdSlr, k, voc = load("bovw.pkl")

# Get the path of the testing images and store them in a list
test_path = 'D:\\Tai-Lieu-Hoc\\TNCKH\\Data_BOVW\\img_data\\test'

testing_names = os.listdir(test_path)

image_paths = []
image_classes = []
class_id = 0


def img_List(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = img_List(dir)
    image_paths += class_path
    image_classes += [class_id]*len(class_path)
    class_id += 1
    

des_list = []

brisk = cv2.BRISK_create(30)

for image_path in image_paths:
    im = cv2.imread(image_path)
    keypoints, des = brisk.detectAndCompute(im, None)
    des_list.append((image_path,des))
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors,descriptor))
    

# Perform vector quantization
from scipy.cluster.vq import vq
# Calculate the histogram of features and represent them as vector
# vq assigns codes from a code book to observations.
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1
        
# Perform Tf-Idf vectorization
nbr_occurences = np.sum((test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences)), "float32")


test_features = stdSlr.transform(test_features)

########################3

true_class = [classes_names[i] for i in image_classes]
predictions = [classes_names[i] for i in clf.predict(test_features)]


print("True_class = " + str(true_class))
print("Prediction = " + str(predictions))


def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()
    
accuracy = accuracy_score(true_class, predictions)
print ("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print (cm)

showconfusionmatrix(cm)


















