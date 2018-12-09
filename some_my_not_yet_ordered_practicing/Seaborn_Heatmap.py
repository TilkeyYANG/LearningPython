# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:27:27 2018
# 主要讲解了FacetGrid参数，适用于总体数据集的子集展示
@author: Invisible-Tilkey
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
np.random.seed(3)
sns.set()

fig1 = plt.figure()
# 生成3X3的随机data
uniform_data = np.random.rand(4, 4)
print(uniform_data)
# 生成heatmap, vmin vmax 可以控制色彩阈值
heatmap = sns.heatmap(uniform_data)
#heatmap = sns.heatmap(uniform_data, vmin=0.01, vmax=1.0)

fig2 = plt.figure()
# randn 随机会出现负数
normal_data = np.random.randn(4, 4)
# 制定一个中心的值 比如是0
ax = sns.heatmap(normal_data, center=0)


'''
pivot 旋转矩阵 超级好用
'''
fig3 = plt.figure()
flights = sns.load_dataset("flights")
flights.head()
# 转换为矩阵形式 横轴为年， 纵轴为月， 数据量为乘客
flights = flights.pivot("month", "year", "passengers")
print (flights)
# format d =十进制整数，annot现实数值
sns.heatmap(flights, annot=True, fmt="d", lw=.05)
fig4 = plt.figure()
sns.heatmap(flights, cmap="YlGnBu", lw=.05)
