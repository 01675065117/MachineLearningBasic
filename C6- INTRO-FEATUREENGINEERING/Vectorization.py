# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:40:55 2021

@author: Admin
"""


import numpy
import cv2
def image_to_vector(image: numpy.ndarray) -> numpy.ndarray:
    """
    Args:
    image: numpy array of shape (length, height, depth)

    Returns:
     v: a vector of shape (length x height x depth, 1)
    """
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))

img = cv2.imread('../Media/image.JPG')

actual = image_to_vector(img)
 
print("vector: {}".format(actual))
length, height, depth = img.shape
expected_shape = (length * height * depth, 1)



















