from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.datasets import mnist, cifar10
from keras.optimizers import SGD
from keras import backend as bk

import keras
import cv2
import numpy as np

def getAlexNetModel(shape_size):

    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=(shape_size, shape_size, 3),
                     padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))


# 修改图像size
def fix_size(x, shape_size):
    x = [cv2.cvtColor(cv2.resize(i, (shape_size, shape_size)), cv2.COLOR_GRAY2BGR) for i in x]
    x = np.concatenate([arr[np.newaxis] for arr in x]).astype('float32')
    return x

## one-hot编码。
def tran_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe

if __name__=="__main__":

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # print(x_train.shape)
    # # 判断是否需要修改通道位次
    # img_rows = 32
    # img_cols = 32
    # if bk.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    #     input_shape = (3, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    #     input_shape = (img_rows, img_cols, 3)
    #
    # # 数据预处理
    # x_train = x_train.astype("float32")
    # x_test = x_test.astype("float32")
    # # 归一化
    # x_train /= 255
    # x_test /= 255
    # # 测试数据转化为向量形式
    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)
    #
    # # 编译模型
    # model = getAlexNetModel(48)
    #
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=["accuracy"])
    #
    # model.summary()
    # # 训练模型
    # batch_size = 128
    # epochs = 10
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test))

    # 超参
    # VGG要求至少48像素
    rshape = 224
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

    # 完全自定义模型
    model_vgg_mnist_pretrain = getAlexNetModel(224)
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
                                 batch_size=10)
    ## 在测试集上评价模型精确度
    scores = model_vgg_mnist_pretrain.evaluate(X_test, y_test_ohe, verbose=0)
    ## 打印精确度
    print(scores)