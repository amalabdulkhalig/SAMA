import keras as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt



plt.style.use('fivethirtyeight')

from keras.datasets import cifar10

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
print(x_train.shape) #(50000 images, 32pixl, 32pixl, 3 RGB colors)
print(y_train.shape) #(50000 rows, 1 label)

print(x_test.shape) #(50000 images, 32pixl, 32pixl, 3 RGB colors)
print(y_test.shape) #(50000 rows, 1 label)