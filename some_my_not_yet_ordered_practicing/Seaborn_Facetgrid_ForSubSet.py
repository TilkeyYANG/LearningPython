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
sns.set(style="dark", color_codes=True)

#titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
#iris = sns.load_dataset("iris")
tips.head()

# 这里的seed只对之后第一次使用random是有效的
np.random.seed(sum(map(ord, "axis_grids")))

fig1 = plt.figure()
# 传入当前数据集， 展示Time =Lunch 和Dinner的两种情况
# 这一次声明先进行占位
g1 = sns.FacetGrid(tips, col="time")
# 柱状图
g1.map(plt.hist, "tip")
g1.add_legend()


fig2 = plt.figure()
# 展示sex的两种情况
g2 = sns.FacetGrid(tips, col="sex", hue="smoker")
# 散点图 alpha 透明程度
g2.map(plt.scatter, "total_bill", "tip", alpha=.6)
# 显示smoker标签
g2.add_legend()


fig3 = plt.figure()
# 两行两列
g3 = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
# fit_reg TF表示回归线
g3.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=True, x_jitter=.05)


from pandas import Categorical
fig4 = plt.figure()
# 用value_counts()获取标签
order_days = tips.day.value_counts().index
print(order_days)
# 也可以自己指定order
order_days = Categorical(['Thur', 'Fri', 'Sat', 'Sun'])
g4 = sns.FacetGrid(tips, row="day", row_order = order_days,
                   size=1.7, aspect=4)
g4.map(sns.boxplot, "total_bill")


 