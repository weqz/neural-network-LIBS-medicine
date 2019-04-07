#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 13:07
# @Author  : ZWQ  
# @File    : 三维图.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import codecs

f = codecs.open("data_pre.txt", 'r', encoding='utf-8')

line = f.readline()

data_list = []
while line:
    a = line.split()

    data_list.append(a)

    line = f.readline()

f.close()

data_list = list(map(np.float32, data_list))

data_1 = data_list[0:110]
data_2 = data_list[110:220]

x_data_1 = []
y_data_1 = []
z_data_1 = []
for data in data_1:
    x = data[0]
    y = data[1]
    z = data[2]
    x_data_1.append(x)
    y_data_1.append(y)
    z_data_1.append(z)

# 创建一个三维
ax = plt.subplot(projection='3d')

ax.scatter(x_data_1, y_data_1, z_data_1, c='r')

x_data_2 = []
y_data_2 = []
z_data_2 = []
for data in data_2:
    x = data[0]
    y = data[1]
    z = data[2]
    x_data_2.append(x)
    y_data_2.append(y)
    z_data_2.append(z)

ax.scatter(z_data_2, y_data_2, z_data_2, c='y')

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')

plt.show()


















