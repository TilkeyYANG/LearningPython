# -*- coding: utf-8 -*cos-
"""
Created on Sun Sep 23 23:38:03 2018

@author: Invisible-Tilkey
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg      =  mnist.train.images
trainlabel    =  mnist.train.labels
testimg       =  mnist.test.images
testlabel     =  mnist.test.labels
print ("MNIST loaded")

# network topologies
# input是像素点个数，classes是最后分类类别（10分类：0~9)，hidden 每层神经元
print ("What does the data of MNIST look like?")
n_input     = 784
n_output    = 10

# network parameters
# w和b变量初始化
stddev = 0.1
weights = {
      # 卷积层权重
      'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev)),
      'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=stddev)),
      # 全链接层权重，其中1024是我们定义的输出向量维数
      'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=stddev)),
      'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=stddev)),
}      
      
# 对w参数一般我们使用高斯初始化
biases = {
      'bc1': tf.Variable(tf.random_normal([64], stddev=stddev)),
      'bc2': tf.Variable(tf.random_normal([128], stddev=stddev)),
      'bd1': tf.Variable(tf.random_normal([1024], stddev=stddev)),
      'bd2': tf.Variable(tf.random_normal([n_output], stddev=stddev)),
}
print ("Network Ready")
# 对b参数用高斯和零值初始化都行

def conv_basic(_input, _w, _b, _keepratio):
  
      # INPUT 对输入做预处理，把输入格式转化（Reshape）为4维的，适应Tensorflow
      # 单张图片处理的情况下，令batch size = 1
      # dim1 = batch size = n // -1表示让Tensorflow自己推算，dim1是可以被自己推算的（另外三维确认情况下）
      # dim2 = height
      # dim3 = width
      # dim4 = channel
      _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])

# =============================================================================
#       # CONV Layer1
# =============================================================================
      # conv2d： 第一个输入是 reshape后输入
      # conv2d： 第二个输入是 权重参数wc1
      # conv2d： 第三个输入是 定义为四维格式，分别对应四个维度的stride/进步大小。一般只更改 h 和 w 的strides其他不变
      # conv2d： 第四个输入是 Padding 'SAME' of 'VALID' 卷积滑动窗口， 
      # SAME 123 234 345 450 500 0填充 || 建议选择
      # VALID 123 234 345 不填充
      _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
      
      # 卷积完后，进行非线性激活函数 也就是ReLU
      _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
      _pool1 = tf.nn.max_pool(_conv1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      # 可以随机杀死一些结点，不参与到总链接， Keepratio是保留参数比如 0.6
      _keepratio = 0.6
      _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)


# =============================================================================
#      # CONV Layer2
# =============================================================================
      _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
      _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
      _pool2 = tf.nn.max_pool(_conv2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

      # VECTORIZE 转换为向量
      # 获取wd1这个向量的形状7*7*128，再将其转换为list得到全链接层需要的向量大小
      _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
      # Fully connected layer 1 全链接层1
      _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
      _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
      # Fully connected layer 2 全链接层2
      _fc2 = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
      
      # RETUEN
      out = { 'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool_dr1': _pool_dr1,
              'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
              'fc1': _fc1, 'fc_dr1': _fc_dr1, 'fc2': _fc2 
              }
      return out
print ("CNN READY")
      
      


# =============================================================================
# 迭代开始！！！！！！
# =============================================================================

#a = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=stddev))
#print (a)
#a = tf.Print(a, [a], "a: ")

# 先占用xy的位置，再一个个batch地往里面进行填充
x = tf.placeholder(tf.float32, [None, n_input]) #None样本个数不确定, n_input=784
y = tf.placeholder(tf.float32, [None, n_output]) #None样本个数不确定, n_class=10
keepratio = tf.placeholder(tf.float32)


# 反向传播 Fonctions
_pred = conv_basic(x, weights, biases, keepratio)['fc2']

# Loss & Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = _pred, labels = y)) #交叉熵函数
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
init = tf.global_variables_initializer()
print ("FONCTIONS READY")

'''
新加入段
'''
# Saver 隔几个Epoch保存一次？
save_step = 1
saver = tf.train.Saver(max_to_keep=3)
# do_train = 1表示训练 0表示读取模型
do_train = 0
sess = tf.Session()
sess.run(init)


print ("GRAPH READY")

# Train Parameter
training_epochs = 15
# batch size 较小，因为运算量太大
batch_size = 16
display_step = 1
    
sess = tf.Session()
sess.run(init)

if do_train == 1:    
  # OPTIMIZE
  for epoch in range(training_epochs):
      avg_cost = 0.
  #    total_batch = int(mnist.train.num_examples/batch_size) # batch次数=总数/batchsize
      total_batch = 10 # 运算量太大，不用全部算完
      #Iteration
      for i in range(total_batch):
          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
          feeds = {x: batch_xs, y: batch_ys, keepratio: 0.7} # batch填充值
          sess.run(optm, feed_dict=feeds)   
          avg_cost += sess.run(cost, feed_dict=feeds)
          
      if epoch % display_step == 0:
          print ("Epoch: %03d/%03d cost:%.9f" % (epoch, training_epochs, avg_cost))
          train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
          print ("TRAIN ACCURACY: %.3f" % (train_acc))
  
      # Save Net
      if epoch % save_step == 0:
          saver.save(sess, "save/nets/cnn_mnist_basic.ckpt-" + str(epoch))
          
  print ("OPTIMIZATION FINISHED")

if do_train == 0:
  epoch = training_epochs-1
  saver.restore(sess, "save/nets/cnn_mnist_basic.ckpt-" + str(epoch))
  
  test_acc = sess.run(accr, feed_dict=feeds)
  print ("TEST ACCURACY: %.3f" % (test_acc))
            
        
        
        
        
        
        
        
        