#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 17:02
# @Author  : ZWQ  
# @File    : label_create.py
# @Software: PyCharm

import numpy as np


data = []

for i in range(105):
    label_1 = [0, 0, 0, 0, 1]
    data.append(label_1)

for i in range(179):
    label_2 = [0, 0, 0, 1, 0]
    data.append(label_2)

for i in range(111):
    label_3 = [0, 0, 1, 0, 0]
    data.append(label_3)

for i in range(205):
    label_4 = [0, 1, 0, 0, 0]
    data.append(label_4)

for i in range(89):
    label_5 = [1, 0, 0, 0, 0]
    data.append(label_5)

data = np.array(data)

print(data.shape)

np.savetxt("label.txt", data)




