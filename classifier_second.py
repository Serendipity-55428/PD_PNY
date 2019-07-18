#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: Pycharm
@file: classifier_second
@time: 2019/7/18 上午9:34
'''
import tensorflow as tf
class Resnet:

    def __init__(self, x, filters, kernel_size, name, padding='same', activation=tf.nn.relu,
                 kernel_initializer=tf.keras.initializers.TruncatedNormal):
        '''
        残差类属性初始化函数
        :param x: 待输入张量, Tensor/Variable
        :param filters: 卷积核个数, int
        :param kernel_size: 卷积核长宽尺寸, list
        :param name: 节点名, str
        :param padding: 标记是否自动补零, str
        :param activation: 激活函数, func
        :param kernel_initializer: 参数初始化函数, func
        '''
        self.__x = x
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__padding = padding
        self.__activation = activation
        self.__kernel_size = kernel_initializer
        self.__name = name

    def resnet_2layers(self):
        '''
        两层卷积的子网络结构
        :return: 子网络残差结构输出
        '''
        conv1 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       activation=self.__activation,
                                       kernel_initializer=self.__kernel_size)(self.__x)
        conv2 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_size)(conv1)
        if self.__x.get_shape().as_list()[-1] != conv2.get_shape().as_list()[-1]:
            x = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_size)(self.__x)
        else:
            x = self.__x
        combination = tf.keras.layers.Add()([conv2, x])
        relu = tf.keras.layers.ReLU(name=self.__name)(combination)
        return relu

    def resnet_3layers(self):
        conv1 = tf.keras.layers.Conv2D(filters=self.__filters // 4,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       activation=self.__activation,
                                       kernel_initializer=self.__kernel_size)(self.__x)
        conv2 = tf.keras.layers.Conv2D(filters=self.__filters // 4,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_size)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_size)(conv2)
        if self.__x.get_shape().as_list()[-1] != conv2.get_shape().as_list()[-1]:
            x = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_size)(self.__x)
        else:
            x = self.__x
        combination = tf.keras.layers.Add()([conv3, x])
        relu = tf.keras.layers.ReLU(name=self.__name)(combination)
        return relu

def subnet_1(x_f, x_l, is_training):
    '''
    三分类残差网络层
    :param x_f: 4个与密度无关特征
    :param x_l: 100个密度特征
    :param is_training: 指示是否在训练
    :return: 神经网络最后输出
    '''
    with tf.name_scope('sub_cnn'):
        x_reshape = tf.reshape(tensor=x_l, shape=[-1, 10, 10, 1], name='x_reshape')
        conv = tf.keras.layers.Conv2D(filters=16, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv1')(x_reshape)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(conv)
        resnet = Resnet(x=pool1, filters=32, kernel_size=[3, 3], name='resnet1')
        res1 = resnet.resnet_2layers()
        resnet2 = Resnet(x=res1, filters=128, kernel_size=[3, 3], name='resnet2')
        res2 = resnet2.resnet_3layers()
        flat = tf.keras.layers.Flatten(name='flat')(res2)
    with tf.name_scope('sub_dnn'):
        x_dnn = tf.concat(values=[flat, x_f], axis=1)
        x_fc1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(x_dnn)
        x_dpt1 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')(inputs=x_fc1, training=is_training)
        x_fc2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(x_dpt1)
        x_dpt2 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')(inputs=x_fc2, training=is_training)
        output = tf.keras.layers.Dense(units=3, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(x_dpt2)
        output2 = tf.keras.activations.softmax(x=output)
        return output2

def subnet_2(x_f, x_l, is_training):
    '''
    11分类残差网络层
    :param x_f: 4个与密度无关特征
    :param x_l: 100个密度特征
    :param is_training: 指示是否在训练
    :return: 神经网络最后输出
    '''
    with tf.name_scope('sub_cnn'):
        x_reshape = tf.reshape(tensor=x_l, shape=[-1, 10, 10, 1], name='x_reshape')
        conv = tf.keras.layers.Conv2D(filters=16, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv1')(x_reshape)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(conv)
        resnet = Resnet(x=pool1, filters=32, kernel_size=[3, 3], name='resnet1')
        res1 = resnet.resnet_2layers()
        resnet2 = Resnet(x=res1, filters=128, kernel_size=[3, 3], name='resnet2')
        res2 = resnet2.resnet_3layers()
        flat = tf.keras.layers.Flatten(name='flat')(res2)
    with tf.name_scope('sub_dnn'):
        x_dnn = tf.concat(values=[flat, x_f], axis=1)
        x_fc1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(x_dnn)
        x_dpt1 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')(inputs=x_fc1, training=is_training)
        x_fc2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(x_dpt1)
        x_dpt2 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')(inputs=x_fc2, training=is_training)
        output = tf.keras.layers.Dense(units=3, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(x_dpt2)
        output2 = tf.keras.activations.softmax(x=output)
        return output2




