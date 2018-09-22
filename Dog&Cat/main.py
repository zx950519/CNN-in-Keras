from keras.applications import *
from keras.preprocessing.image import *
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np


def write_gap(MODEL, image_size, lambda_func=None):

    # 构建输入层
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    # 根据参数 填充对应的预设卷积层
    if lambda_func:
        x = Lambda(lambda_func)(x)
    # 基础模型
    base_model = MODEL(input_tensor=x,
                       weights="imagenet",
                       include_top=False)
    # GlobalAveragePooling2D 将卷积层输出的每个激活图直接求平均值
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    # 从指定目录下批量生成增强数据
    train_generator = gen.flow_from_directory("./imgs/train",
                                              image_size,
                                              shuffle=False,
                                              batch_size=16)
    test_generator = gen.flow_from_directory("./imgs/test",
                                             image_size,
                                             shuffle=False,
                                             batch_size=16,
                                             class_mode=None)
    # 为来自数据生成器的输入样本生成预测
    train = model.predict_generator(train_generator, train_generator.samples)
    test = model.predict_generator(test_generator, test_generator.samples)
    #
    with h5py.File("gap_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

if __name__ == "__main__":

    # 提取特征向量
    # write_gap(ResNet50, (224, 224))
    # write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
    # write_gap(Xception, (299, 299), xception.preprocess_input)

    import h5py
    import numpy as np
    from sklearn.utils import shuffle

    np.random.seed(2017)

    X_train = []
    X_test = []

    for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    X_train, y_train = shuffle(X_train, y_train)

    from keras.models import *
    from keras.layers import *

    np.random.seed(2017)

    input_tensor = Input(X_train.shape[1:])
    x = Dropout(0.5)(input_tensor)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, x)

    model.compile(optimizer='adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2)

    # np.random.seed(2017)
    #
    # # 载入特征向量
    # X_train = []
    # X_test = []
    # for filename in ["feature_vector_ResNet50.h5",
    #                  "feature_vector_Xception.h5",
    #                  "feature_vector_InceptionV3.h5"]:
    #     with h5py.File(filename, 'r') as h:
    #         X_train.append(np.array(h['train']))
    #         X_test.append(np.array(h['test']))
    #         y_train = np.array(h['label'])
    #
    # X_train = np.concatenate(X_train, axis=0)
    # X_test = np.concatenate(X_test, axis=0)
    #
    # X_train, y_train = shuffle(X_train, y_train)
    #
    # from keras.models import *
    # from keras.layers import *
    #
    # np.random.seed(2017)
    #
    # input_tensor = Input(X_train.shape[1:])
    # x = Dropout(0.5)(input_tensor)
    # x = Dense(1, activation='sigmoid')(x)
    # model = Model(input_tensor, x)
    #
    # model.compile(optimizer='adadelta',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #
    # model.fit(X_train, y_train, batch_size=128, epochs=8, validation_split=0.2)

