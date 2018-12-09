# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%
import tensorflow as tf
# TF库与Numpy库类似 尽量多用float32

#声明tf下的Variable
num = tf.Variable(0)


#调用tf的add函数进行递加
new_num = tf.add(num, tf.constant(1))

#调用tf的assign函数进行赋值
update = tf.assign(num, new_num)

#打开Session计算块
with tf.Session() as sess:

    #初始化/执行 全球变量
    sess.run(tf.global_variables_initializer())
    
    #通过sess.run（变量）进行输出
    print(sess.run(num))
    
    #循环结
    for _ in range(3):
        sess.run(update)
        print (sess.run(num))
        
#保存当前session的方式
    #save_path = saver.save(sess, "D://tensorflow//model//test180918")
    #%%