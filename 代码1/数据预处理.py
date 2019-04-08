#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 9:54
# @Author  : ZWQ  
# @File    : 数据预处理.py
# @Software: PyCharm

import numpy as np
import codecs
import os

from sklearn.decomposition import PCA


def readOneFolder(filename_1, filename_2, column):
    """
    读取两个文件夹filename_1和filename_2中的所有文件files，
    并把file文件中指定的列数据转换成数组list
    把所有的files文件读取出的lists合并到一个二维矩阵中
    :param filename: 文件路径
    :param column: 要读取的列
    :return: 只包含文件夹某一列的二维矩阵
    """
    files_1 = os.listdir(filename_1)    # files_1是一个数组
    files_2 = os.listdir(filename_2)

    # 返回的数据存储在data中
    data = []

    # 读取第一个文件夹中所有文件的第column列中的内容添加到data[]中
    for file in files_1:

        # 文件相对路径
        path = filename_1 + '/' + file

        # 打开文件
        f = codecs.open(path, mode='r', encoding='utf-8')

        # 读取文件的第一行, 一次只能读一行
        line = f.readline()

        list_1 = []

        # 把f中的某一列读取出来，存储到列表中
        while line:
            a = line.split()
            b = a[column]
            list_1.append(b)
            line = f.readline()
        f.close()

        # array[]是一个字符串列表，转化成实数列表
        list_2 = list(map(np.float32, list_1))

        data.append(list_2)

    # 读取第二个文件夹中所有文件的第column列的内容添加到data[]中
    for file in files_2:

        # 文件相对路径
        path = filename_2 + '/' + file

        # 打开文件
        f = codecs.open(path, mode='r', encoding='utf-8')

        # 读取文件的第一行, 一次只能读一行
        line = f.readline()

        list_1 = []

        # 把f中的某一列读取出来，存储到列表中
        while line:
            a = line.split()
            b = a[column]
            list_1.append(b)
            line = f.readline()
        f.close()

        # array[]是一个字符串列表，转化成实数列表
        list_2 = list(map(np.float32, list_1))

        data.append(list_2)

    # 把data转化成array
    data = np.array(data)
    print(data.shape)

    return data


def pca(data, n_components):
    """
    主成分分析进行特征降维（数据会改变，特征数量也会减少）
    :param data: 输入的原始数据
    :param n_components: 降维后数据信息保留百分比（0~1）
    :return: 降维之后的数据
    """
    pca = PCA(n_components=n_components)
    data_re =pca.fit_transform(data)

    print(data_re.shape)

    return data_re


if __name__ == "__main__":

    data = readOneFolder("DH", "QH", 1)
    data_re = pca(data=data, n_components=0.999)

    np.savetxt("data_pre.txt", data_re)









































