# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:32:25 2022

@author: Khoa
"""

import cv2
import numpy as np
import os

train_path = 'D:\\Tai-Lieu-Hoc\\TNCKH\\Data_BOVW\\img_data\\train'

training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0


def imgList(path):
    return [os.path.join(path,f) for f in os.listdir(path)]



for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imgList(dir)
    image_paths += class_path
    image_classes += [class_id]*len(class_path)
    class_id += 1
    
'''
- Create feature extraction and keypoint detector objects
- Decription of feature extraction, keypoint detector and BRISK https://codelungtung.wordpress.com/2018/09/28/keypoint-detector-local-features-and-fast/
- Create list where all the descriptors will be stored
'''
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
    
# Kmeans works only on float, so convert integers to float
descriptors_float = descriptors.astype(float)

# Perform kmeans clustering and vector quantization
from scipy.cluster.vq import kmeans, vq
k = 200
voc, variance = kmeans(descriptors_float,k,1)

# Calculate the histogram of features and represent them as vector
# vq assigns codes from a code book to observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
        
# Perform Tf-Idf vectorization
nbr_occurences = np.sum((im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences)), "float32")
















