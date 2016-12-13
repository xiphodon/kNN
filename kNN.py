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

def kNN_demo1():
    '''
    k-近邻算法模拟数据与实现分类
    :return:
    '''
    group, labels = createDataSet()
    print(classify0([1.0, 0.9], group, labels, 3))


def file2matrix(filename):
    '''
    将文本记录转换为矩阵
    [文件数据为各个网友的信息，信息每列分别为：
        1，每年获得的飞行常客里程数
        2，玩视频游戏所耗时间百分比
        3，每周消费的冰淇淋的公升数
        4，魅力值
    ]
    :param filename: 数据文件路径
    :return: 数据矩阵，类型标签集合
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines) # 获取文件行数
    returnMat = np.zeros((numberOfLines,3)) # 对应数据转化为矩阵
    classLabelVector = [] # 类型标签集合
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(like_str2int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


def like_str2int(like):
    '''
    魅力值字符串转数字
    :param like: 魅力值程度字符串
    :return: 魅力值程度对应的数字
    '''
    if like == 'largeDoses':
        return 3
    elif like == 'smallDoses':
        return 2
    elif like == 'didntLike':
        return 1
    else:
        return 0


def kNN_demo2():
    '''
    k-近邻算法改进约会网站的配对效果
    :return:
    '''
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    print(datingDataMat,datingLabels)

if __name__ == '__main__':
    # kNN_demo1()
    kNN_demo2()