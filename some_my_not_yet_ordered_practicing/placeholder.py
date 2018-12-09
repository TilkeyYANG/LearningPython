# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%
import tensorflow as tf

#占用两个未知量
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
#运算未知量
output = tf.multiply(input1, input2)

#Session运算
with tf.Session() as sess:
    #赋值给input 1，2
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
    #%%