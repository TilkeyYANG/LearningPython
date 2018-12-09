# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%
import tensorflow as tf
a = tf.constant('hello TF')
sess = tf.Session()
print(sess.run(a))
#%%