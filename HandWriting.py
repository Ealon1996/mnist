# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 02:41:11 2018

@author: s7856
"""

import numpy as np 
from keras.utils import np_utils
from keras.datasets import mnist 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print (X_test.shape)
