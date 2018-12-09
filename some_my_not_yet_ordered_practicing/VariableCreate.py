# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%
import tensorflow as tf
# TF库与Numpy库类似   尽量多用float32

# switch Matrix to tf
x = tf.Variable([[0.5,1.0]]) 
y = tf.Variable([[2.0],[1.0]])

# Matrix Multi 矩阵乘法
z = tf.matmul(x, y)
 
# the form of z
print(z) 

#initializing variables 全局变量初始化
init_op = tf.global_variables_initializer() 

with tf.Session() as sess:
    sess.run(init_op)
    print (z.eval())
    
    
    
    #%%