# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:20:15 2021

@author: Harsh
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 11:38:35 2021

@author: Harsh
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('Initial shape training set:')
print(x_train.shape, y_train.shape)

#preprocessing
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
num_classes = 10
#y_train = keras.utils.to_categorical(y_train,num_classes)
#y_test = keras.utils.to_categorical(y_test,num_classes)

input_shape = (28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('Training set shape',x_train.shape)
print('No. of training sample size:',x_train.shape[0])
print('No. of test sample size:',x_test.shape[0])

#model
model = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=input_shape, padding='same'), 
    keras.layers.AveragePooling2D(), 
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
    keras.layers.AveragePooling2D(), 
    keras.layers.Flatten(), 
    keras.layers.Dense(120, activation='tanh'), 
    keras.layers.Dense(84, activation='tanh'), 
    keras.layers.Dense(10, activation='softmax') 
])

model.compile(optimizer = keras.optimizers.Adam(),
              loss = keras.losses.sparse_categorical_crossentropy,
              metrics = ['accuracy'])

hist = model.fit(x_train, y_train, epochs = 20,  verbose = 1, validation_data = (x_test,y_test))
print('Training complete')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model_hdlenet5.h5')
print('model saved')






























