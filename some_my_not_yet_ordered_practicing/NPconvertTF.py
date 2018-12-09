# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%
import tensorflow as tf
import numpy as np
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))
    #%%