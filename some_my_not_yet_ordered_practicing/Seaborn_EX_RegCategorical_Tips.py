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
pal = sns.cubehelix_palette(8, start=-.75, rot=.75)
sns.set_palette(pal)

np.random.seed(sum(map(ord, "categorical")))
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# =============================================================================
# TIPS 
tips.head()
# =============================================================================

# =============================================================================
# # STRIPPLOT 不太推荐的散点图（视觉容易堆叠）
# =============================================================================
fig1 = plt.figure()
fig1.add_subplot(211)
sns.stripplot(x="day", y="total_bill", data=tips)
# =============================================================================
# # STRIPPLOT 加入抖动，分布更清晰
# =============================================================================
fig1.add_subplot(212)
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)

# =============================================================================
# # swarmplot 树形抖动，同时用色彩区分sex标签
# =============================================================================
fig2 = plt.figure()
sns.swarmplot(x="total_bill", y="day", hue="sex" , data=tips)

# =============================================================================
# # BOXplot 盒图 用于判断离群点 可以判断数据整体分布
# =============================================================================
fig3 = plt.figure()
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips)

# =============================================================================
# # Violinplot 提琴图  # split True时，可以把左右分割
# =============================================================================
fig4 = plt.figure()
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)

# =============================================================================
# # Violinplot + Swarmplot 套用
# =============================================================================
fig5 = plt.figure()
sns.violinplot(x="total_bill", y="day", hue="sex", data=tips, split=True)
sns.swarmplot(x="total_bill", y="day", hue="time" , data=tips)

# =============================================================================
# TITANICS
titanics.head()
# =============================================================================
