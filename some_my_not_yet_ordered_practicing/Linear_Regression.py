# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##创造一组近似线性的数据

# 随机生成1000个点，在y=0.1x+0.3周围
pts = 1000
vectors_set = []

for i in range(pts):
    x1 = np.random.normal(0.0, 0.55)
# np.random.normal 高斯分布的均值和标准差
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

# 生成样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# 显示关于 x y 的点阵
plt.scatter(x_data, y_data, c='r')
plt.show()


#=================================================

##构建线性回归模型

# 生成斜率1维矩阵，取值是[-1,1]
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='w')
# 生成1维矩阵，取值是0
b = tf.Variable(tf.zeros([1]), name='b')

# 经过计算得出预估值y
y = w * x_data + b

# 以预估值y和实际值y_data之间的 均方差 作为损失 || 迭代优化
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 最小化 均方差 来进行训练
train = optimizer.minimize(loss, name='train')


#**注解：用在最末调用20次train
#**每一次train对对象optimizer 以train类（0.5）参数的梯度下降法 对loss参数进行minimize（）
#**总之 每次w和b都会被更优化一些

# run init session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# 第一次赋值的w和b
print ("w =", sess.run(w), "b = ", sess.run(b), "loss = ", sess.run(loss))
# 20 train
for step in range(20):
    sess.run(train)
    # 输出训练好的w和b
    print ("w =", sess.run(w), "b = ", sess.run(b), "loss = ", sess.run(loss))       
#

#%%
