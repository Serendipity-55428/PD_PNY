#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: garner
@file: hole_model
@time: 2019/7/17 下午8:45
'''
import tensorflow as tf
import numpy as np
import os
from Saving_model_pb import SaveImport_model, use_pb, import_ops

def check(first_check):
    '''
    整体分类器
    :param first_check: 初级分类器
    :return: 初级分类器, func
    '''
    dict1 = {0:7.17, 1:17.93, 2:20}
    dict2 = {0:0.05, 1:3.26, 2:4.48, 3:23, 4:35, 5:71, 6:107, 7:143, 8:179, 9:215, 10:251}
    def second_check1():
        '''
        第二层分类器1
        :return: 具体类别
        '''
        nonlocal  dict1
        pb_file_path = os.getcwd()
        g2 = tf.Graph()
        with g2.as_default():
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g2) as sess:
            sess.run(init)
            use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/second_classifier1')
            input = np.arange(107) #改
            x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
            x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
            is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
            output = import_ops(sess_new=sess, op_name='dnn/Softmax')
            r_classify = sess.run(output, feed_dict={x_f: input[:4][np.newaxis, :], x_l: input[4:-3][np.newaxis, :],
                                                     is_training: False})
            r_classify = r_classify.argmax()
            r_finally = dict1[r_classify]
        return r_finally

    def second_check2():
        '''
        第二层分类器1
        :return: 具体类别
        '''
        nonlocal dict2
        pb_file_path = os.getcwd()
        g3 = tf.Graph()
        with g3.as_default():
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g3) as sess:
            sess.run(init)
            use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/second_classifier2')
            input = np.arange(107)  # 改
            x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
            x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
            is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
            output = import_ops(sess_new=sess, op_name='dnn/Softmax')
            r_classify = sess.run(output, feed_dict={x_f: input[:4][np.newaxis, :], x_l: input[4:-3][np.newaxis, :],
                                                     is_training: False})
            r_classify = r_classify.argmax()
            r_finally = dict2[r_classify]
        return r_finally

    r_classify = first_check()
    if r_classify == 0:
        r_finally = 0.01
    elif r_classify == 1:
        r_finally = second_check1()
    else:
        r_finally = second_check2()
    print('最优半径为: %s' % r_finally)
    return first_check

@check
def first_check():
    '''
    第一层分类器
    :return: 半径所属大类别, int
    '''
    pb_file_path = os.getcwd()
    g1 = tf.Graph()
    with g1.as_default():
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g1) as sess:
        sess.run(init)
        use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/first_classifier')
        input = np.arange(107) #改
        x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
        x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
        is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
        output = import_ops(sess_new=sess, op_name='dnn/Softmax')
        r_classify = sess.run(output, feed_dict={x_f: input[:4][np.newaxis, :], x_l: input[4:-3][np.newaxis, :],
                                                 is_training: False})
        r_classify = r_classify.argmax()
    return r_classify

if __name__ == '__main__':
    pass






