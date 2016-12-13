#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/13 15:08
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : kNN.py
# @Software: PyCharm

# k-近邻算法

import numpy as np
import operator # 位运算符模块

def createDataSet():
    '''
    创建数据集合
    :return: 数据集合，对应标签
    '''
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]) # 创建数组
    labels = ['A', 'A', 'B', 'B'] # 数组中每个元素数组对应的标签
    return group, labels