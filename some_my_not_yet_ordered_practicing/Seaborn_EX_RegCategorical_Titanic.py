# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:27:27 2018

@author: Invisible-Tilkey
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
pal = sns.cubehelix_palette(8, start=-.5, rot=.75)
sns.set_palette(pal)

np.random.seed(sum(map(ord, "categorical")))
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# =============================================================================
# TITANICS
titanic.head()
# =============================================================================

# =============================================================================
# # BARPLOT
# =============================================================================
fig1 = plt.figure()
fig1.add_subplot(121)
sns.barplot(x="class", y="survived", hue="sex", data=titanic)
# =============================================================================
# # SPOINTPLOT 点图 可以描述差异性
# # 可以使用markers和linestyles参数
# =============================================================================
fig1.add_subplot(122)
sns.pointplot(x="class", y="survived", hue="sex", data=titanic, 
              markers=["^", "*"], linestyles=["-", "--"])

# =============================================================================
# # boxplot 横画
# =============================================================================
fig2 = plt.figure()
sns.boxplot(data=iris, orient="h")

# =============================================================================
# # BOXplot 盒图 用于判断离群点 可以判断数据整体分布
# =============================================================================
fig3 = plt.figure()
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips)

# =============================================================================
# # Factorplot 多层面板分析, 不指定为折现，指定的话kind=""
# =============================================================================
fig4 = plt.figure()
sns.factorplot(x="day", y="total_bill", hue="sex", data=tips, kind="swarm", size=6)

# =============================================================================
# # 拆分Factorplot #aspect 长宽比 #size 大小
# =============================================================================
fig5 = plt.figure()
sns.factorplot(x="time", y="total_bill", hue="sex", data=tips,
               col="day", kind="swarm", size=3, aspect=.5)
