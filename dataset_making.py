#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: garner
@file: dataset_making
@time: 2019/7/15 下午4:09
'''
import numpy as np
import pandas as pd
from collections import Counter
from inputdata import LoadFile, SaveFile
from DataRead import checkclassifier

def making(nums_cl, dataset):
    '''
    按照输入的每个半径的分组以及数据量个数划分数据集
    :param nums_cl: 每个半径数据的数量以及划分为哪一类，输入时按照最优半径从大到小排列 type=[[nums,cl], [,], ...]
    :param dataset: 数据集/最后一列为标签
    :return: 经过处理后的数据集,数据集最后一列为标签
    '''
    dataset_output = np.zeros(shape=[1, dataset.shape[-1]])
    #最优半径从小到大排序
    dict_r = Counter(dataset[:, -1])
    # print(dict_r)
    dataset_2 = dict_r.keys()
    # print(dataset_2)
    dataset_2 = np.array(list(dataset_2))
    # print(dataset_2)
    r_sort = np.sort(dataset_2)
    dataset_pd = pd.DataFrame(data=dataset, columns=['f'+'%s' % i for i in range(dataset.shape[-1]-1)]+['label'])
    i = 0
    for r in r_sort:
        print('正在执行半径%s' % r)
        #定义用于拼接的临时矩阵
        data_per_pd = dataset_pd.loc[dataset_pd['label'] == r]
        data_per = np.array(data_per_pd)
        np.random.shuffle(data_per)
        #获取该最优半径所需要的数据量和标签值
        nums, cl = nums_cl[i]
        data_per = np.hstack((data_per[:nums, :-1], np.ones(dtype=np.float32, shape=[nums, 1])*cl))
        dataset_output = data_per if dataset_output.any() == 0 else np.vstack((dataset_output, data_per))
        i += 1
    return dataset_output

if __name__ == '__main__':
    p = r'/home/xiaosong/桌面/PNY_all.pickle'
    dataset = LoadFile(p)
    nums_cl = [[5100, 0], [310, 2], [3, 2], [213, 2], [2030, 1], [2030, 1], [1030, 1], [1000, 2], [691, 2], [766, 2],
               [892, 2], [552, 2], [271, 2], [160, 2], [213, 2]]
    dataset_output = making(nums_cl=nums_cl, dataset=dataset)
    print(dataset_output.shape)
    checkclassifier(dataset_output[:, -1])
    # SaveFile(dataset_output, savepickle_p=r'/home/xiaosong/桌面/PNY_3cl.pickle')