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
    # 创建数组，每个元素视为每条数据的两个特征
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B'] # 数组中每个元素数组对应的标签
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    k-近邻算法
    :param inX: 用于分类的输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return: 预测数据所在分类
    '''
    dataSetSize = dataSet.shape[0] # 获取dataSet矩阵行数，即向量条目数

    # 使目标向量形成与dataSet矩阵相同行数矩阵，对应相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 # 矩阵各个元素平方
    sqDistances = sqDiffMat.sum(axis=1) # 一个维度上求和
    distances = sqDistances ** 0.5 # 开方
    # 以上四步为计算目标向量与每个向量间的距离【d=[(x1 - x2)^2 + (y1 - y2)^2]^0.5】

    sortedDistIndicies = distances.argsort() # 距离排序好的索引值
    classCount = {}
    # 选取距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] # 按距离排好序的每个向量对应的标签值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 统计各个标签出现的次数

    # 把字典转换为元组对的列表，按照第2个域值（即标签出现的次数）逆序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回统计后距离最小的k个点中出现次数最多的标签
    return sortedClassCount[0][0]



if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([1.0, 0.9],group,labels,3))