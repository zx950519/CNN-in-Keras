from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import os
import keras

# 超参
# 批处理量
batch_size = 32
# 类别数
num_classes = 10
# 迭代次数
epochs = 10
# 是否采用数据增强
data_augmentation = True
# 未知
num_predictions = 20
# 模型存储位置
save_dir = os.path.join(os.getcwd(), 'saved_models')
# 权重存储未知
model_name = 'keras_cifar10_trained_model.h5'

def getZFNetModel():

    model = Sequential()

    model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=x_train.shape[1:], padding='valid', activation='relu',
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

# ## one-hot编码。
# def tran_y(y):
#     y_ohe = np.zeros(10)
#     y_ohe[y] = 1
#     return y_ohe
#
# # 修改图像size
# def fix_size(x, shape_size):
#     x = [cv2.cvtColor(cv2.resize(i, (shape_size, shape_size)), cv2.COLOR_GRAY2BGR) for i in x]
#     x = np.concatenate([arr[np.newaxis] for arr in x]).astype('float32')
#     return x
#
# def train():
#     # 超参
#     # VGG要求至少48像素
#     rshape = 48
#     # 训练轮次
#     epoch = 10
#     # 批处理大小
#     batch_size = 64
#
#     # 加载数据
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     X_train = fix_size(X_train, rshape)
#     X_test = fix_size(X_test, rshape)
#     # 归一化
#     X_train /= 255.0
#     X_test /= 255.0
#     # 去one-hot处理
#     y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
#     y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
#     y_train_ohe = y_train_ohe.astype('float32')
#     y_test_ohe = y_test_ohe.astype('float32')
#
#     # 完全自定义模型
#     model_vgg_mnist_pretrain = getZFNetModel()
#     # 自定义优化器
#     sgd = SGD(lr=0.05, decay=1e-5)
#     # 模型编译
#     model_vgg_mnist_pretrain.compile(loss='categorical_crossentropy',
#                                      optimizer=sgd,
#                                      metrics=['accuracy'])
#     # 模型训练
#     model_vgg_mnist_pretrain.fit(X_train,
#                                  y_train_ohe,
#                                  validation_data=(X_test, y_test_ohe),
#                                  epochs=epoch,
#                                  batch_size=128)
#     ## 在测试集上评价模型精确度
#     scores = model_vgg_mnist_pretrain.evaluate(X_test, y_test_ohe, verbose=0)
#     ## 打印精确度
#     print(scores)
#
# if __name__=="__main__":
#    train()

if __name__=="__main__":

    # 加载数据
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('训练集张量:', x_train.shape)
    print(x_train.shape[0], '个训练样例')
    print(x_test.shape[0], '个测试样例')

    # 转化为one-hot形式
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 获得自定义模型
    model = getZFNetModel()

    # 采用RMSprop作为优化器
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # 模型编译
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # 数据归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # 判断是否使用数据增强
    if not data_augmentation:
        print("未使用数据增强")
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print("使用数据增强")
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)

    # 保存模型以及权重
    json_string = model.to_json()
    open("ZFNet_model.json", "w").write(json_string)
    model.save_weights("ZFNet_weights.h5")

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])