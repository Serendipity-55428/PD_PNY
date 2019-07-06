#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: graph
@time: 2019/6/30 下午12:59
'''
import tensorflow as tf
import numpy as np
from inputdata import LoadFile, SaveFile, onehot, input, guiyi, spliting
from classifier import layers
def session(dataset_path, train_path, test_path):
    '''
    节点连接
    :param dataset_path: 数据集路径
    :param train_path: 训练集数据路径
    :param test_path: 测试集数据路径
    :return: None
    '''
    #导入数据集
    dataset = LoadFile(p=dataset_path)
    dataset = guiyi(dataset)
    dataset = onehot(dataset)
    # training_set = LoadFile(p=train_path)
    # testing_set = LoadFile(p=test_path)
    # #特征归一化
    # training_set = guiyi(training_set)
    # testing_set = guiyi(testing_set)
    # #onehot(后15维是稀疏01编码)
    # training_set = onehot(training_set)
    # testing_set = onehot(testing_set)

    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('placeholder'):
            x_f = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='x_f')
            x_l = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='x_l')
            y = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='y')
            learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        output = layers(x_f=x_f, x_l=x_l, is_training=is_training)
        with tf.name_scope('prediction'):
            # loss = -tf.reduce_mean(y * tf.log(output), name='loss')
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=output))
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.keras.backend.argmax(output, axis=1),
                                                   tf.keras.backend.argmax(y, axis=1)), tf.float32), name='pred')
        with tf.name_scope('etc'):
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g) as sess:
        sess.run(init)
        for i in range(10000):
            for data in input(dataset=dataset, batch_size=1000):
                _ = sess.run(opt, feed_dict={x_f:data[:, :4], x_l:data[:, 4:-4], y:data[:, -4:],
                                             learning_rate:1e-4, is_training:True})
            if i % 100 == 0:
                loss_ = sess.run(loss, feed_dict={x_f:data[:, :4], x_l:data[:, 4:-4], y:data[:, -4:],
                                                  is_training:True})
                testing_set = spliting(dataset, size=2000)
                acc_ = sess.run(acc, feed_dict={x_f:testing_set[:, :4], x_l:testing_set[:, 4:-4],
                                                    y:testing_set[:, -4:], is_training:False})
                print('第%s轮训练集损失函数值为: %s  测试集准确率为: %s' % (i, loss_, acc_))

if __name__ == '__main__':
    p = r'/home/xiaosong/桌面/PNY_fft_dadta_bigclass.pickle'
    p1 = r'/home/xiaosong/桌面/PNY_train.pickle'
    p2 = r'/home/xiaosong/桌面/PNY_test.pickle'
    session(dataset_path=p, train_path=p1, test_path=p2)


