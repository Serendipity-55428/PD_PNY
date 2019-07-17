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

pb_file_path = os.getcwd()
g1 = tf.Graph()
with g1.as_default():
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config= tf.ConfigProto(gpu_options= gpu_options), graph= g1) as sess:
    sess.run(init)
    use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/first_classifier')
    input = np.arange(107)
    x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
    x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
    is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
    output = import_ops(sess_new=sess, op_name='dnn/Softmax')
    r = sess.run(output, feed_dict={x_f:input[:4][np.newaxis, :], x_l:input[4:-3][np.newaxis, :], is_training:False})
    print(r)


