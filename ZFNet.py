from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist, cifar10

import numpy as np
import cv2

def getZFNetModel(shape_size):

    model = Sequential()

    model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=(shape_size, shape_size, 3), padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

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

    # 完全自定义模型
    model_vgg_mnist_pretrain = getZFNetModel(48)
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