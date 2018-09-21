from PIL import Image
from keras.models import model_from_json

import os
import cv2
import gc
import numpy as np

# 修改图像size
def fix_size(x, shape_size):
    x = [cv2.cvtColor(cv2.resize(i, (shape_size, shape_size)), cv2.COLOR_GRAY2BGR) for i in x]
    x = np.concatenate([arr[np.newaxis] for arr in x]).astype('float32')
    return x

def resize(input_path, output_path, width, height, type):
    """
    调整指定图片的尺寸
    :param input_path:输入的图片路径
    :param output_path:输出的图片路径
    :param width:输出图片的宽度
    :param height:输出图片的高度
    :param type:输出图片类型
    """
    img = Image.open(input_path)
    out = img.resize((width, height), Image.ANTIALIAS)
    out.save(output_path, type)

def save_model(model, model_output_path, weight_output_path, overwrite):
    """
    保存模型及网络权重
    :param model:模型
    :param model_output_path:模型输出路径
    :param weight_output_path:权重输出路径
    :param overwrite:是否覆写
    :return:
    """
    if(os.path.exists(model_output_path)):
        if(overwrite==False):
            print("已有模型")
            return
    if (os.path.exists(weight_output_path)):
        if (overwrite == False):
            print("已有权重")
            return
    json_string = model.to_json()
    open(model_output_path, "w").write(json_string)
    model.save_weights(weight_output_path)

    print("保存完毕")

def load_exist_model(model_output_path, weight_output_path):
    """
    加载模型及网络权重
    :param model_output_path:
    :param weight_output_path:
    :return:
    """
    if (os.path.exists(model_output_path)==False):
        print("没有模型")
        return
    else:
        model = model_from_json(open(model_output_path).read())

    if (os.path.exists(weight_output_path)):
        print("没有权重")
        return model
    else:
        model.load_weights(weight_output_path)
        return model

# 调整学习率
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)

    return lr


# 垃圾回收
def garbage_collection(x):
    for i in range(x):
        gc.collect()