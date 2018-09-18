from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


import keras
import os

def getLeNetModel():
    # 序贯模型
    model = Sequential()
    # 卷积层-1
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     input_shape=(28, 28, 1), padding="valid",
                     activation="relu", kernel_initializer="uniform"))
    # 池化层-1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 卷积层-2
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                     input_shape=(28, 28, 1), padding="valid",
                     activation="relu", kernel_initializer="uniform"))
    # 池化层-2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 展开层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(100, activation="relu"))
    # 全连接层
    model.add(Dense(10, activation="softmax"))
    # 返回模型
    return model




