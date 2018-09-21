from keras.datasets import cifar10
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, BatchNormalization, Input, Activation
from keras.layers import Convolution2D, GlobalAveragePooling2D, Concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.models import model_from_json
from keras.applications import *
from utils import *
from keras.callbacks import ModelCheckpoint

import keras.backend as K
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 数据增强
def getDataGenerator(train_phase,rescale=1./255):
    if train_phase == True:
        datagen = ImageDataGenerator(
        rotation_range=0.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        channel_shift_range=0.,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=rescale)
    else:
        datagen = ImageDataGenerator(
        rescale=rescale
        )
    return datagen

# 卷积块
def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    # 激活层
    x = Activation('relu')(input)
    # 卷积层
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    # Dropout层
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

# 全连接块
def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter

# Transition块
def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    # 卷积层
    x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
    # Dropout层
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    # 池化层
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    # 标准化层
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    return x

# 创建DenseNet
def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):
    # 输入层
    model_input = Input(shape=img_dim)
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    # Depth must be 3 N + 4
    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    # 每个全连接块中的层数
    nb_layers = int((depth - 4) / 3)

    # 卷积层
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)
    # 标准化层
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)

    # 添加全连接块
    print("总共有："+str(nb_dense_block)+"添加全连接块")
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # 添加Transition块
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    # 最后一个全连接块没有Transition块
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    # 激活层
    x = Activation('relu')(x)
    # 池化层
    x = GlobalAveragePooling2D()(x)
    # 全连接层
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    # 模型汇总
    densenet = Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

def fit(ROWS, COLS, CHANNELS, nb_classes, batch_size, modelPath, weightsPath, overwrite):

    # 删除旧模型已经权重
    if (os.path.exists(modelPath)):
        if (overwrite == True):
            os.remove(modelPath)
    if (os.path.exists(weightsPath)):
        if (overwrite == True):
            os.remove(weightsPath)

    # 图像格式
    img_dim = (ROWS, COLS, CHANNELS)
    # 网络深度
    densenet_depth = 40
    # ?
    densenet_growth_rate = 12

    # 加载数据
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 数据归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # 类别由单列转化为向量
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    # 数据增强
    train_datagen = getDataGenerator(train_phase=True)
    train_datagen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_datagen = getDataGenerator(train_phase=False)
    validation_datagen = validation_datagen.flow(x_test, y_test, batch_size=batch_size)

    # 建立模型
    model = createDenseNet(nb_classes=nb_classes, img_dim=img_dim, depth=densenet_depth,
                           growth_rate=densenet_growth_rate)
    # 是否加载预训练权重
    # if resume == True:
    #     model.load_weights(check_point_file)
    # 自定义优化器
    optimizer = Adam()
    # optimizer = SGD(lr=0.001)

    # 模型编译
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # 模型概览
    model.summary()

    # 训练模型-方式1
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=20,
              verbose=1,
              validation_data=(x_test, y_test))

    # 训练模型-方式2
    # check_point_file = filepath="./model/checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5"
    # model_checkpoint = ModelCheckpoint(check_point_file, monitor="val_acc", save_best_only=True,
    #                                    save_weights_only=True, verbose=1)
    # callbacks = [model_checkpoint]
    # model.fit_generator(generator=train_datagen,
    #                 steps_per_epoch= x_train.shape[0] // batch_size,
    #                 epochs=10,
    #                 callbacks=callbacks,
    #                 validation_data=validation_datagen,
    #                 validation_steps = x_test.shape[0] // batch_size,
    #                 verbose=1)

    # 保存模型及权重
    json_string = model.to_json()
    open(modelPath, "w").write(json_string)
    model.save_weights(weightsPath)

def train():
    # 行
    ROWS = 32
    # 列
    COLS = 32
    # 通道
    CHANNELS = 3
    # 类别总数
    nb_classes = 10
    # 每批处理的数据量
    batch_size = 128
    # 是否覆盖原模型、权重
    overwrite = True
    # 训练模型
    fit(ROWS, COLS, CHANNELS, nb_classes, batch_size,
                "./model/LeNet_model.json", "./model/LeNet_weights.h5", overwrite)

def test(modelPath, weightsPath, overwrite):
    if (os.path.exists(modelPath)):
        if (overwrite == True):
            os.remove(modelPath)
    else:
        print("模型不存在")
        return
    if (os.path.exists(weightsPath)):
        if (overwrite == True):
            os.remove(weightsPath)
    else:
        print("模型不存在")
        return
    # 加载模型权重
    model = model_from_json(open(modelPath).read())
    model.load_weights(weightsPath)

    # Todo 验证


shape_size = 32     # 图像大小
channel = 3    # 通道
nb_classes = 10     # 类别总数
epoch = 10      # 迭代次数
batch_size = 128    # 每批处理的数据量
pre_weight_file_path = ""   # 预训练权重文件路径
model_save_path = "./VGGNet_model.json"    # 模型保存位置
weight_save_path = "./VGGNet_weights.h5"   # 权重保存位置
diy = True         # 自定义网格或采用Keras预设
data_augmentation = False   # 是否使用数据增强
pre_weight = False  # 是否加载预训练权重

if __name__ == "__main__":

    # 图像格式
    img_dim = (shape_size, shape_size, channel)
    # 网络深度
    densenet_depth = 40
    # ?
    densenet_growth_rate = 12

    # 加载数据
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 数据归一化
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # 类别由单列转化为向量
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    # 数据增强
    train_datagen = getDataGenerator(train_phase=True)
    train_datagen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    validation_datagen = getDataGenerator(train_phase=False)
    validation_datagen = validation_datagen.flow(x_test, y_test, batch_size=batch_size)

    if diy == True:
        # 建立模型
        model = createDenseNet(nb_classes=nb_classes, img_dim=img_dim, depth=densenet_depth,
                               growth_rate=densenet_growth_rate)
        # 是否加载预训练权重
        if pre_weight == True:
            model.load_weights(pre_weight_file_path)
        # 自定义优化器
        optimizer = Adam()
        # optimizer = SGD(lr=0.001)
        # 模型编译
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # 模型概览
        model.summary()
    else:
        model = DenseNet121(include_top=True, weights="imagenet")
        # 自定义优化器
        optimizer = Adam()
        # optimizer = SGD(lr=0.001)
        # 模型编译
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # 模型概览
        model.summary()

    # 设置模型保存路径
    filepath = ""
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

    if data_augmentation == True:

        # check_point_file = filepath="./model/checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5"
        # model_checkpoint = ModelCheckpoint(check_point_file, monitor="val_acc", save_best_only=True,
        #                                    save_weights_only=True, verbose=1)
        # callbacks = [model_checkpoint]
        # model.fit_generator(generator=train_datagen,
        #                 steps_per_epoch=x_train.shape[0] // batch_size,
        #                 epochs=10,
        #                 callbacks=callbacks,
        #                 validation_data=validation_datagen,
        #                 validation_steps = x_test.shape[0]// batch_size,
        #                 verbose=1)

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
                            epochs=epoch, verbose=1, workers=4,
                            callbacks=callbacks)
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=20,
                  verbose=1,
                  validation_data=(x_test, y_test))

    # 保存模型及权重
    json_string = model.to_json()
    open(model_save_path, "w").write(json_string)
    model.save_weights(weight_save_path)


