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


model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

adam = Adam(lr=1e-4)

# show model

model.summary()


model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')

model.fit(X_train, y_train, epochs=10, batch_size=64)

print('\nTesting ------------')

loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

model.save('HandWrite_model.h5')
