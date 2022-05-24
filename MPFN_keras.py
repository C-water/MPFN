from keras import layers
from keras.models import Model
import math
from keras import backend as K
from core_layers import *
from keras.applications.vgg16 import VGG16

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time


def conv_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1, strides=(1,1)):
    if dilation_rate == 0:
        y = layers.Conv2D(filters,1,padding=padding,use_bias=use_bias,
            activation='relu')(x)
    else:
        y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides,
            activation='relu')(x)
    return y

def conv(x, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
    y = layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    return y

def network_v10_new1_91113(nFilters, multi=True):
    conv_func = conv_relu

    def pre_block(x, d_list):
        t = x
        for i in range(len(d_list)):
            _t = conv_func(t, nFilters, 3, dilation_rate=d_list[i])
            t = layers.Concatenate(axis=-1)([_t, t])
        a = conv_func(t, nFilters * 4, 1, use_bias=False)
        a = conv_func(a, nFilters * 2, 1, use_bias=False)
        a = conv_func(a, 1, 1, use_bias=False)
        a = Sign()(a)
        _a = layers.Lambda(lambda x: (1 - x))(a)
        t = conv(t, nFilters * 2, 1)
        _t = layers.Multiply()([t, a])
        _x = layers.Multiply()([x, _a])
        t = layers.Add()([_x, _t])
        return t

    def pos_block(x, d_list):
        t = x
        for i in range(len(d_list)):
            _t = conv_func(t, nFilters, 3, dilation_rate=d_list[i])
            t = layers.Concatenate(axis=-1)([_t, t])
        t = conv_func(t, nFilters * 2, 1)
        return t

    def global_block(x):
        t = layers.ZeroPadding2D(padding=(1, 1))(x)
        t = conv_func(t, nFilters * 4, 3, strides=(2, 2))
        t = layers.GlobalAveragePooling2D()(t)
        # t = layers.Dense(nFilters*16, activation='relu')(t)
        t = layers.Dense(nFilters * 8, activation='relu')(t)
        t = layers.Dense(nFilters * 4)(t)
        _t = conv_func(x, nFilters * 4, 1)
        _t = layers.Multiply()([_t, t])
        _t = conv_func(_t, nFilters * 2, 1)
        return _t

    def filter_block(x, f, window):
        # f: the feature map used to generate filters
        # x: the image to be filtered
        f = conv(f, window ** 2, 3, use_bias=False)
        f = layers.Activation('softmax')(f)
        y = FilterLayer(window=window)([x, f])
        return y

    output_list = []
    d_list_a = (1, 1, 1, 1, 1)
    d_list_b = (1, 1, 1, 1, 1)
    d_list_c = (1, 1, 1, 1, 1)
    x = layers.Input(shape=(None, None, 5))  # 16m*16m
    x_2 = layers.Input(shape=(None, None, 3))

    _x = Space2Depth(scale=2)(x)
    t1 = conv_func(_x, nFilters * 2, 3, padding='same')  # 8m*8m
    t1 = pre_block(t1, d_list_a)
    t2 = layers.ZeroPadding2D(padding=(1, 1))(t1)
    t2 = conv_func(t2, nFilters * 2, 3, padding='valid', strides=(2, 2))  # 4m*4m
    t2 = pre_block(t2, d_list_b)

    t3 = layers.ZeroPadding2D(padding=(1, 1))(t2)
    t3 = conv_func(t3, nFilters * 2, 3, padding='valid', strides=(2, 2))  # 2m*2m
    t3 = pre_block(t3, d_list_c)
    t3 = pre_block(t3, d_list_c)
    _t3 = conv(t3, 12, 3)
    _t3 = Depth2Space(scale=2)(_t3)
    _t3 = layers.Add()([_t3, x_2])

    t3 = pos_block(t3, d_list_c)
    t3 = conv_func(t3, nFilters * 4, 1)
    t3 = Depth2Space(scale=2)(t3)
    t3_out = filter_block(_t3, t3, 13)  # 4m*4m
    t3_out_up = layers.UpSampling2D(size=(2,2),interpolation='bilinear')(t3_out)
    output_list.append(t3_out)
    t2 = layers.Concatenate()([t3_out, t2])
    t2 = conv_func(t2, nFilters * 2, 1)
    t2 = pre_block(t2, d_list_b)
    _t2 = conv(t2, 12, 3)
    _t2 = Depth2Space(scale=2)(_t2)
    _t2 = layers.Add()([_t2, t3_out_up])

    t2 = pos_block(t2, d_list_b)
    t2 = conv_func(t2, nFilters * 4, 1)
    t2 = Depth2Space(scale=2)(t2)
    t2_out = filter_block(_t2, t2, 11)  # 8m*8m
    t2_out_up = layers.UpSampling2D(size=(2,2),interpolation='bilinear')(t2_out)
    output_list.append(t2_out)
    t1 = layers.Concatenate()([t1, t2_out])
    t1 = conv_func(t1, nFilters * 2, 1)
    t1 = pre_block(t1, d_list_a)
    _t1 = conv(t1, 12, 3)
    _t1 = Depth2Space(scale=2)(_t1)
    _t1 = layers.Add()([_t1, t2_out_up])
    t1 = pos_block(t1, d_list_b)
    t1 = conv_func(t1, nFilters * 4, 1)
    t1 = Depth2Space(scale=2)(t1)
    y = filter_block(_t1, t1, 9)  # 16m*16m
    output_list.append(y)
    if multi != True:
        return models.Model([x,x_2], y)
    else:
        return models.Model([x,x_2], output_list)


if __name__ == "__main__":
    # model = network_EDSR()
    model = network_v10_new1_91113(64,multi=True)
    model.summary()
