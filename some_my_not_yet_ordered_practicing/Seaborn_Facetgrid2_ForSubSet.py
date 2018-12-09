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

tips = sns.load_dataset("tips")
tips.head()
iris = sns.load_dataset("iris")
iris.head()


fig1 = plt.figure()
# 创建颜色字典
pal = dict(Lunch="seagreen", Dinner="gray")
g1 = sns.FacetGrid(tips, hue="time", palette=pal, size=5,
                   hue_kws={"marker": ["*", "v"]})
# s=点的大小 linewidth=描边大小 edgecolor=描边颜色
g1.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, 
       linewidth=.5, edgecolor="orange")
g1.add_legend()


fig2 = plt.figure()
with sns.axes_style("white"):
  g2 = sns.FacetGrid(tips, row="sex", col="smoker", 
                    margin_titles=True, size=2.5)
g2.map(plt.scatter, "total_bill", "tip", edgecolor="orange", lw=.5)
"""
# 设置轴的名称("Xname","Yname")，取值范围xticks yticks,可以设置w/hspace子图间隔
"""
g2.set_axis_labels("Total Bill", "Tip")
g2.set(xticks=[10, 30, 50], yticks=[2, 6, 10])
g2.fig.subplots_adjust(wspace=.02, hspace=.02)

"""
# 直接指定对图以后, 可以在对角线上柱形图非对角上散点
"""
fig3 = plt.figure()
g3 = sns.PairGrid(iris, hue="species", 
                  palette=sns.cubehelix_palette(8, start=-.75, rot=.75))
g3.map_diag(plt.hist)
g3.map_offdiag(plt.scatter, edgecolor="gray", lw=.2, alpha=.7, s=30)
g3.add_legend()

"""
# 输出指定的参数图
"""
fig4 = plt.figure()
g4 = sns.PairGrid(tips, hue="size", 
                  palette="GnBu_d")
g4.map_diag(plt.hist)
g4.map_offdiag(plt.scatter, edgecolor="gray", lw=.2, alpha=.7, s=30)
g4.add_legend()
# =============================================================================
# 
# from pandas import Categorical
# fig4 = plt.figure()
# # 用value_counts()获取标签
# order_days = tips.day.value_counts().index
# print(order_days)
# # 也可以自己指定order
# order_days = Categorical(['Thur', 'Fri', 'Sat', 'Sun'])
# g4.map(sns.boxplot, "total_bill")
# 
# 
#  
# =============================================================================
