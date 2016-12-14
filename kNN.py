#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/12/13 15:08
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : kNN.py
# @Software: PyCharm

# k-近邻算法

import numpy as np
import operator  # 位运算符模块


def createDataSet():
    '''
    创建数据集合
    :return: 数据集合，对应标签
    '''
    # 创建数组，每个元素视为每条数据的两个特征
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # 数组中每个元素数组对应的标签
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
    dataSetSize = dataSet.shape[0]  # 获取dataSet矩阵行数，即向量条目数

    # 使目标向量形成与dataSet矩阵相同行数矩阵，对应相减
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # 矩阵各个元素平方
    sqDistances = sqDiffMat.sum(axis=1)  # 一个维度上求和
    distances = sqDistances ** 0.5  # 开方
    # 以上四步为计算目标向量与每个向量间的距离【d=[(x1 - x2)^2 + (y1 - y2)^2]^0.5】

    sortedDistIndicies = distances.argsort()  # 距离排序好的索引值
    classCount = {}
    # 选取距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 按距离排好序的每个向量对应的标签值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计各个标签出现的次数

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
    numberOfLines = len(arrayOLines)  # 获取文件行数
    returnMat = np.zeros((numberOfLines, 3))  # 对应数据转化为矩阵
    classLabelVector = []  # 类型标签集合
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(like_str2int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def like_str2int(like):
    '''
    魅力值字符串转数字
    :param like: 魅力值程度字符串
    :return: 魅力值程度对应的数字
    '''
    if like.isdigit():
        return int(like)
    elif like == 'largeDoses':
        return 3
    elif like == 'smallDoses':
        return 2
    elif like == 'didntLike':
        return 1
    else:
        return 0


def show_window(datingDataMat, datingLabels):
    '''
    使用matplotlib，按标签集来显示不同颜色的“玩视频游戏所耗时间百分比”和“每周所消费的冰淇淋公升数”
    :param datingDataMat: 数据矩阵
    :return:
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    '''
    数据归一化，使所有数据映射在[0,1]区间内
    {原理：newData = (oldData - min)/(max - min)}
    :param dataSet: 原数据矩阵
    :return: 归一化特征值后的矩阵，
    '''
    minVals = dataSet.min(0) # 从列中选取最小值 得到1*3行向量
    maxVals = dataSet.max(0) # 从列中选取最小值 得到1*3行向量
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1)) # oldData - min
    normDataSet = normDataSet/np.tile(ranges, (m,1))   # (oldData - min)/(max - min)
    return normDataSet, ranges, minVals


def datingClassTest():
    '''
    数据分类测试
    :return:
    '''
    hoRatio = 0.10 # 取用10%的数据
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio) # 测试数据选用数量
    errorCount = 0.0 # 错误计数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print("划分分类为: %d, 真实分类为: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("分类错误率为: %3.2f" % (100*errorCount/float(numTestVecs)) + "%")
    print("错误数: %d" % (errorCount))


def classifyPerson():
    '''
    分类约会网友
    :return: 分类结果
    '''
    resultList = [u'不喜欢',u'有点喜欢',u'很喜欢']
    ffMiles = float(input("每年飞行的里程数:"))
    percentTats = float(input("玩视频游戏所耗的时间百分比:"))
    iceCream = float(input("每周消费的冰淇淋公升数:"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("你对这个人的喜欢程度为：", resultList[classifierResult - 1])


def kNN_demo2():
    '''
    k-近邻算法改进约会网站的配对效果
    :return:
    '''
    # datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    # print(datingDataMat, datingLabels)
    # show_window(datingDataMat, datingLabels)
    # normMat,ranges,minVals = autoNorm(datingDataMat)
    # print(normMat,ranges,minVals)
    datingClassTest()
    classifyPerson()


def img2vector(filename):
    '''
    将32*32的二进制图像矩阵转换为1*1024的行向量
    :param filename: 二进制图像矩阵存储文件路径
    :return: 1*1024的行向量
    '''
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    '''
    手写数字识别系统测试
    :return:
    '''
    import os
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits') # 获取目录所有文件名
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] # 去掉.txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits') # 获取目录所有文件名
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0] # 去掉 .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if (classifierResult != classNumStr):
            print("分类预测结果为: %d, 真实结果为: %d" % (classifierResult, classNumStr))
            errorCount += 1.0
    print("\n分类错误数量为: %d" % errorCount)
    print("\n分类错误率: %3.2f" % (100*errorCount/float(mTest)) + "%")


def kNN_demo3():
    '''
    k-近邻算法实现手写识别系统
    :return:
    '''
    # imgArray = img2vector('digits/trainingDigits/0_2.txt')
    # print(imgArray)
    handwritingClassTest()

if __name__ == '__main__':
    # kNN_demo1()
    # kNN_demo2()
    kNN_demo3()
