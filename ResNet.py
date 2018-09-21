from __future__ import print_function

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from utils import *
from keras.applications import *
from matplotlib import pyplot as plt

import numpy as np
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# 不同深度模型的参数
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

# 定义一个resnet层
# 预激活结构：BN-ReLU-conv
# 后激活结构：conv-BN-ReLU
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',  # He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，
                  # 其中fan_in权重张量的扇入（输入单元数目）
                  kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:  # 后激活（post activation）
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:  # 预激活(pre_activation)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# 构建残差网络resnet_v1
# feature map sizes:
# conv1: 32x32, 16
# stage 0: 32x32, 16
# stage 1: 16x16, 32
# stage 2:  8x8,  64
# 每一个stage，都有n个残差模块(如 n=3 )
# 每一个残差模块均采用：conv(3*3)->conv(3*3)结构
# after-addition activation: 采用ReLU
# Skip Connection：
# stage 0: identiy-> identity-> identity
# stage 1: conv-> identity-> identity
# stage 2: conv-> identity-> identity

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    # 网络第一层conv(32x32)->BN->ReLU
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            if stack > 0 and res_block == 0:
                strides = 2
            strides = 1
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2  # 下采样后，filters 数目增倍
    # 增加分类层
    # v1 在averagepooling 层前不使用BN
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # 实例化模型
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 构建残差网络resnet_v2
# Features maps sizes:
# conv1  : 32x32,  16
# stage 0: 32x32,  64
# stage 1: 16x16, 128
# stage 2:  8x8,  256
# 每一个stage 的都有n个残差模块 (如 n=3 )
# 每一个残差模块均采用 bottleneck 结构：conv(1x1)->conv(3x3)->conv(1x1)
# after-addition activation: identity mapping
# Skip Connection :
# stage 0: conv-> identity-> identity
# stage 1: conv-> identity-> identity
# stage 2: conv-> identity-> identity

def resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)
    # 第一个3x3卷积层之后执行 BN-ReLU ，然后再 splitting into 2 paths,进入第一个残差模块
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

                    # bottleneck residual 结构
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out
    # 增加分类层
    # v2 在在 Pooling前执行 BN-ReLU
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # 实例化模型
    model = Model(inputs=inputs, outputs=outputs)
    return model


batch_size = 32     # paper中设置 batch_size=128
epochs = 50    # 训练轮次
diy = True         # 自定义网格或采用Keras预设
data_augmentation = False    # 是否采用数据增强
num_classes = 10    # 类别数
subtract_pixel_mean = True  # 是否对矩阵进行求均值->做差
version = 2     # 模型型号选择，version = 1 (ResNet v1) version = 2 (ResNet v2)
n = 3   # 设置模型层次深度

if __name__=="__main__":

    # 环境设置
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # 选择模型版本
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # 模型名
    model_type = 'ResNet%dv%d' % (depth, version)

    # 加载data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 数据格式(32*32*3)
    input_shape = x_train.shape[1:]
    # 数据预处理
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # 将标签转化为one-hot编码
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if diy == True:
        # 模型编译
        if version == 2:
            model = resnet_v2(input_shape=input_shape, depth=depth)
        else:
            model = resnet_v1(input_shape=input_shape, depth=depth)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        # 模型概览
        model.summary()
        # 画出模型结构图，并保存成图片
        plot_model(model, to_file="ResNetV2.png")
    else:
        # 灵活设置
        model = ResNet50(include_top=True, weights="imagenet",)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])
        # 模型概览
        model.summary()

    # 设置模型保存路径
    save_dir = os.path.join(os.getcwd(), "ResNet_models")
    model_name = "resnet_cifar10_%s_model.{epoch:03d}.h5" % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # 设置回调函数,用于中间模型保存和学习率调整,仅保存最优模型
    checkpoint = ModelCheckpoint(filepath=filepath,
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

    # 数据增强+模型训练
    if not data_augmentation:
        print("不使用数据增强")
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print("使用实时数据增强")
        # This will do preprocessing and realtime data augmentation:
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
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks)

    # 模型评估
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
