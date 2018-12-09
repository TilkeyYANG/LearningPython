# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 22:47:08 2018

@author: Invisible-Tilkey
"""
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# color_palette() 能传入 Matplotlib色
# set_palette() 设置所有图颜色

# 打印默认色板
# 默认六个 deep muted pastel bright dark colorblind
current_palette = sns.color_palette()
sns.palplot(current_palette)

# hls的默认颜色空间 8色
sns.palplot(sns.color_palette("hls", 8))

# hls的默认颜色空间 12色
sns.palplot(sns.color_palette("hls", 4))


fig = plt.figure()
# 试打印 20个值 8列
boxdata = np.random.normal(size=(20, 8)) + np.arange(8) / 2
boxfig = sns.boxplot(data = boxdata, palette=sns.color_palette("hls", 8))
sns.set_context("notebook", font_scale = 0.4, rc={"lines.linewidth": 2.5}) 

# 调整亮度 l=7， s=2
sns.palplot(sns.hls_palette(12, l=.7, s=.65))

# 双色相近
sns.palplot(sns.color_palette("Paired", 8))

# 连续渐变色板
sns.palplot(sns.color_palette("Blues", 8))
# 翻转渐变 _r
sns.palplot(sns.color_palette("BuGn_r", 8))

# 线性色板（l/s线性） cubehelix
sns.palplot(sns.color_palette("cubehelix", 8))
# 线性色板（l/s线性） cubehelix
sns.palplot(sns.cubehelix_palette(8, start=.75, rot=-.150))
sns.palplot(sns.cubehelix_palette(8, start=.5, rot=.75))
sns.palplot(sns.cubehelix_palette(8, start=-.5, rot=.75))

# 指定深潜色板
sns.palplot(sns.light_palette("brown", 8))
# reverse 翻转色斑 dark从暗开始
sns.palplot(sns.dark_palette("pink", 8, reverse = True))

figcmap = plt.figure()
# 等高线生成实例
x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
pal = sns.light_palette((240, 90, 60), input="husl", as_cmap=True)
sns.kdeplot(x, y, cmap=pal)