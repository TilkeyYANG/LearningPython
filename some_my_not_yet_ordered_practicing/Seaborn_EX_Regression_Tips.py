# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 00:15:33 2018

@author: Invisible-Tilkey
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "regression")))
# 下载消费数据集 Tips
tips = sns.load_dataset("tips")
# 直接显示信息
tips.head()

# regplot # 可lmplot函数但是规范更复杂
# 试探 总金额 和 小费 的关系
sns.regplot(x="total_bill", y="tip", data=tips)

fig2 = plt.figure()
# 试探 人数size 和 小费 的关系
fig2.add_subplot(211)
sns.regplot(x="size", y="tip", data=tips)
# 加入浮动 因为 size 太直 不适合回归分析
fig2.add_subplot(212)
sns.regplot(x="size", y="tip", data=tips, x_jitter = 0.2)

