# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:45:09 2018

@author: Invisible-Tilkey
"""
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

# 单变量 单特征 分析

fig1 = plt.figure()
# 生成高斯数据
x = np.random.normal(size=30)
# 生成直方图， kde = 密度估计 false
sns.distplot(x, kde=False)
# 过于密集，手动指定bins=20个小块
sns.distplot(x, bins=20, kde=False)
sns.distplot(x, bins=60, kde=False)

fig2 = plt.figure()
# 生成gamma数据
y = np.random.gamma(6, size=30)
# 勾勒轮廓 这里fit表示轮廓线类型
sns.distplot(y, kde=False, fit=stats.gamma)

fig3 = plt.figure()
# 生成均值与协方差
mean, cov = [0, 1], [(1, 0.5), (0.5, 1)]
data = np.random.multivariate_normal(mean, cov, 100) 
df = pd.DataFrame(data, columns=["x", "y"] )
# 此时输入df可以输出
# JOINTPLOT散点图 并且同时画出了直方图
sns.jointplot(x="x", y="y", data=df)

# JOINTPLOT散点图进阶 HEX
c, d = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
  sns.jointplot(x=c, y=d, kind="hex", color="k" ) # k=黑色图


# 鸢尾花数据集
iris = sns.load_dataset("iris")
# PAIRPLOT把一组数据所有两两制图
sns.pairplot(iris)

