from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras import backend as bk
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from utils import *


import cv2
import numpy as np
import gc
import keras
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

shape_size = 48     # 图像大小
epoch = 10          # 训练轮次
batch_size = 32     # 批处理大小
diy = False         # 自定义网格或采用Keras预设
data_augmentation = False   # 是否使用数据增强
num_classes = 10    # 分类数
checkpoint_path = "./model/vggnet_minist_model.{epoch:03d}.h5"  # 回调函数记忆点的保存位置
model_save_path = "./VGGNet_model.json"    # 模型保存位置
weight_save_path = "./VGGNet_weights.h5"   # 权重保存位置

if __name__ == "__main__":
    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 灰度图->RGB彩图 与LeNet中重构张量执行类似作用
    x_train = fix_size(x_train, shape_size)
    x_test = fix_size(x_test, shape_size)

    # 数据预处理
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # 归一化
    x_train /= 255
    x_test /= 255

    # one-hot处理
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(y_train.shape)

    garbage_collection(10)

    if diy == True:
        print("自定义模型")
        model = getVGGNetModel(48)
        # 自定义优化器
        sgd = SGD(lr=0.05, decay=1e-5)
        # 模型编译
        model.compile(loss='categorical_crossentropy',
                                         optimizer=sgd,
                                         metrics=['accuracy'])
        model.summary()

    else:
        print("系统模型+全连接层")
        model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(shape_size, shape_size, 3))

        for layer in model_vgg.layers:
            layer.trainable = False
        # 添加最后的几层
        model_fin = Flatten()(model_vgg.output)
        model_fin = Dense(4096, activation='relu', name='fc1')(model_fin)
        model_fin = Dense(4096, activation='relu', name='fc2')(model_fin)
        model_fin = Dropout(0.5)(model_fin)
        model_fin = Dense(10, activation='softmax', name='prediction')(model_fin)

        # 最终的网络
        model = Model(model_vgg.input, model_fin, name='vgg16_pretrain')
        # 自定义优化器
        sgd = SGD(lr=0.05, decay=1e-5)
        # 模型编译
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        # 网络概览
        model.summary()

    # 设置回调函数,用于中间模型保存和学习率调整,仅保存最优模型
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    # 学习率调度器
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    if data_augmentation == True:
        print("使用实时数据增强")
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epoch,
                            verbose=1,
                            workers=4,
                            callbacks=callbacks)
    else:
        print("不使用数据增强")
        model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs=epoch,
                  batch_size=batch_size)

    # 统计得分
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # 保存模型及权重
    json_string = model.to_json()
    open(model_save_path, "w").write(json_string)
    model.save_weights(weight_save_path)