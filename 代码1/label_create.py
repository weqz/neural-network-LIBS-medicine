#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/7 11:30
# @Author  : ZWQ  
# @File    : label_create.py
# @Software: PyCharm

import numpy as np

data = []
for i in range(110):
    list1 = [0, 1]
    data.append(list1)

for i in range(110):
    list1 = [1, 0]
    data.append(list1)

data = np.array(data)
print(data.shape)

np.savetxt("label.txt", data)



