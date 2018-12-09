# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 20:04:37 2018

@author: Invisible-Tilkey
"""

import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# matplotlib inline

def sinplot(flip=1):
  # 在坐标0~14上找出100个点
  x = np.linspace(0, 14, 100)
  for i in range(1, 7):
    plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
    

# 定义一个figure
fig = plt.figure()

# nrow, ncol, number
fig.add_subplot(221)
# 使用sns默认的set()
sns.set()
sinplot()
# 去掉上有框线
sns.despine()

# 只隐藏左轴, 离轴线距离
sns.despine(left=True, offset=5)


fig.add_subplot(222)
# 默认的五种主题风格
# darkgrid | whitegrid grid 只有刻度线 | dark 深色，无线 | white | ticks 加入刻度线
sns.set_style("whitegrid")
boxdata = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data = boxdata)

# With内会全部参照style
with sns.axes_style("darkgrid"):
  plt.subplot(223)
  sinplot()
plt.subplot(224)
sinplot(-1)
 
# context style = notebook, talk, paper; font_scale = 3.5 ie; rc={""}
sns.set_context("notebook", font_scale = 2.5, rc={"lines.linewidth": 2.5}) 
# 坐标轴
plt.figure(figsize=(8, 6))
sinplot()