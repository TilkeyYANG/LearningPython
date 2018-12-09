# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:22:19 2018
保存模型
@author: Invisible-Tilkey
"""
import tensorflow as tf

# =============================================================================
# 把v1 v2 通过 saver 保存
# =============================================================================
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(init_op)
  print ("V1: ", sess.run(v1))
  print ("V2: ", sess.run(v2))
  saver_path = saver.save(sess, "D:\ML\Savers\SaverTest\model.ckpt")
# 保存了计算域
  print("Model saved in file: ", saver_path)

# =============================================================================
# 读取v1 v2
# =============================================================================

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "D:\ML\Savers\SaverTest\model.ckpt")
  print ("V1: ", sess.run(v1))
  print ("V2: ", sess.run(v2))
  print ("Model restored")

