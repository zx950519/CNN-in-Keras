from __future__ import print_function
from keras import backend as bk
from keras.datasets import mnist
from keras.models import model_from_json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import LeNet5 as lenet5
import AlexNet as alexnet
import VGGNet as vggnet
import ZFNet as zxnet
import GoogleNet as googlenet
import DenseNet as densenet
import ResNet as resnet
import keras
import numpy as np
import cv2


def fit(model, batchSize, numClass, epoch, img_row, img_col, modelSavePath, weightSavePath, overwrite):
    """
    :param model: 调用获取模型的函数
    :param batchSize:每批处理多少条数据
    :param numClass:结果分为多少类
    :param epoch:迭代轮次
    :param img_row:图像大小
    :param img_col:图像大小
    :param modelSavePath:模型保存路径
    :param weightSavePath:权重保存路径
    :param overwrite:是否覆写
    """

    if (os.path.exists(modelSavePath)):
        if (overwrite == True):
            os.remove(modelSavePath)
    if (os.path.exists(weightSavePath)):
        if (overwrite == True):
            os.remove(weightSavePath)

    # 单批处理数目
    batch_size = batchSize
    # 种类大小
    num_classes = numClass
    # 迭代轮次
    epochs = epoch
    # 图像的size
    img_rows = img_row
    img_cols = img_col

    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 判断是否需要修改通道位次
    if bk.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 数据预处理
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # 归一化
    x_train /= 255
    x_test /= 255
    # 测试数据转化为向量形式
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 编译模型
    # model = lenet5.getLeNetModel()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])
    # 训练模型
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    # 统计得分
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # 保存模型及权重
    json_string = model.to_json()
    open(modelSavePath, "w").write(json_string)
    model.save_weights(weightSavePath)

def predit(modelPath, weightPath, imgPath):
    model = model_from_json(open(modelPath).read())
    model.load_weights(weightPath)
    img = imgPath
    testData = np.empty((1, 1, 28, 28), dtype="float32")
    imgData = cv2.imread(img, 0)
    arr = np.asarray(imgData, dtype="float32")
    testData[0, :, :, :] = arr
    testData = testData.reshape(testData.shape[0], 28, 28, 1)
    return model.predict_classes(testData, batch_size=1, verbose=0)

if __name__ == "__main__":
    # 训练LeNet并测试
    # fit(lenet5.getLeNetModel(), 128, 10, 10, 28, 28, "./model/LeNet_model.json", "./model/LeNet_weights.h5", True)
    # print(predit("./model/LeNet_model.json", "./model/LeNet_weights.h5", "./data/test/6/mnist_test_81.png"))

    # 训练AlNet并测试
    # fit(alexnet.getAlexNetModel(), 128, 10, 10, 28, 28, "./model/AlexNet_model.json", "./model/AlexNet_weights.h5", True)
    # print(predit("./model/AlexNet_model.json", "./model/AlexNet_weights.h5", "./data/test/6/mnist_test_81.png"))

    # 训练ZFNet并测试
    # fit(zxnet.getZFNetModel(), 128, 10, 10, 28, 28, "./model/ZFNet_model.json", "./model/ZFNet_weights.h5",
    #     True)
    # print(predit("./model/ZFNet_model.json", "./model/ZFNet_weights.h5", "./data/test/6/mnist_test_81.png"))

    # 训练VGGNet并测试
    # fit(vggnet.getVGGNetModel(), 128, 10, 10, 28, 28, "./model/VGGNet_model.json", "./model/VGGNet_weights.h5",
    #     True)
    # print(predit("./model/VGGNet_model.json", "./model/VGGNet_weights.h5", "./data/test/6/mnist_test_81.png"))

    # 训练GoogleNet并测试-目前有问题
    # fit(googlenet.create_googlenet(), 128, 10, 10, 28, 28, "./model/GoogleNet_model.json", "./model/GoogleNet_weights.h5",
    #     True)
    # print(predit("./model/GoogleNet_model.json", "./model/GoogleNet_weights.h5", "./data/test/6/mnist_test_81.png"))

    # 训练ResNet并测试
    # fit(resnet.getResNet(), 128, 10, 10, 28, 28, "./model/ResNet_model.json", "./model/ResNet_weights.h5",
    #     True)
    # print(predit("./model/ResNet_model.json", "./model/ResNet_weights.h5", "./data/test/6/mnist_test_81.png"))

    # 训练DenseNet并测试 需要224*224*3的训练集 但是目前只有28*28*1的训练集
    # model = keras.applications.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    # fit(model, 128, 10, 10, 28, 28, "./model/DenseNet_model.json", "./model/DenseNet_weights.h5",
    #     True)
    # print(predit("./model/DenseNet_model.json", "./model/DenseNet_weights.h5", "./data/test/6/mnist_test_81.png"))