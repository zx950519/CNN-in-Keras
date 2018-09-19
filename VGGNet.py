from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD

import cv2
import numpy as np
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def getVGGNetModel(shape_size):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(shape_size, shape_size, 3), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

## one-hot编码。
def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

# 修改图像size
def fix_size(x, shape_size):
    x = [cv2.cvtColor(cv2.resize(i, (shape_size, shape_size)), cv2.COLOR_GRAY2BGR) for i in x]
    x = np.concatenate([arr[np.newaxis] for arr in x]).astype('float32')
    return x

# 回收资源
def garbage_collection(x):
    for i in range(x):
        gc.collect()

def train():
    # 超参
    # VGG要求至少48像素
    rshape = 48
    # 训练轮次
    epoch = 10
    # 批处理大小
    batch_size = 64

    # 加载数据
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = fix_size(X_train, rshape)
    X_test = fix_size(X_test, rshape)
    # 归一化
    X_train /= 255.0
    X_test /= 255.0
    # 去one-hot处理
    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
    y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
    y_train_ohe = y_train_ohe.astype('float32')
    y_test_ohe = y_test_ohe.astype('float32')

    # VGG16 全参重训迁移学习
    # 很多时候需要多次回收垃圾才能彻底收回内存。如果不行，重新启动单独执行下面的模型
    garbage_collection(10)

    # # 系统模型+自定义部分组件
    # model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(rshape, rshape, 3))
    #
    # for layer in model_vgg.layers:
    #     layer.trainable = False
    # # 添加最后的几层
    # model = Flatten()(model_vgg.output)
    # model = Dense(4096, activation='relu', name='fc1')(model)
    # model = Dense(4096, activation='relu', name='fc2')(model)
    # model = Dropout(0.5)(model)
    # model = Dense(10, activation='softmax', name='prediction')(model)
    #
    # # 最终的网络
    # model_vgg_mnist_pretrain = Model(model_vgg.input, model, name='vgg16_pretrain')
    # # 网络概览
    # model_vgg_mnist_pretrain.summary()

    # 完全自定义模型
    model_vgg_mnist_pretrain = getVGGNetModel(48)
    # 自定义优化器
    sgd = SGD(lr=0.05, decay=1e-5)
    # 模型编译
    model_vgg_mnist_pretrain.compile(loss='categorical_crossentropy',
                                     optimizer=sgd,
                                     metrics=['accuracy'])
    # 模型训练
    model_vgg_mnist_pretrain.fit(X_train,
                                 y_train_ohe,
                                 validation_data=(X_test, y_test_ohe),
                                 epochs=epoch,
                                 batch_size=128)
    ## 在测试集上评价模型精确度
    scores = model_vgg_mnist_pretrain.evaluate(X_test, y_test_ohe, verbose=0)
    ## 打印精确度
    print(scores)

if __name__=="__main__":
   train()