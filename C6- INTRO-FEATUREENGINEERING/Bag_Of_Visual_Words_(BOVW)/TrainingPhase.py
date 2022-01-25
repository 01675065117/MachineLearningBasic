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












