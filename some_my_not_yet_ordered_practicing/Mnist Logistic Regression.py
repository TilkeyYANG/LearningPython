# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:22:04 2018

@author: Invisible-Tilkey
"""

#%%


#** 输入几个data库

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data

print ("packs loaded")

#** 如果有0~9这几个数值，则0的0=true，0的9=false
mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg      =  mnist.train.images
trainlabel    =  mnist.train.labels
testimg       =  mnist.test.images
testlabel     =  mnist.test.labels
print ("MNIST loaded")

print (" type of 'mnist' is %s" % (type(mnist)))
print (" nb of train data is %d" % (mnist.train.num_examples))
print (" nb of test data is %d" % (mnist.test.num_examples))

print (trainimg.shape)
print (trainlabel.shape)
print (testimg.shape)
print (testlabel.shape)

print (trainlabel[0])
#** 载入一个Training Data：
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

# None表示不知道有多少样本，784是像素数量

# 自变量的784个像素
x = tf.placeholder("float", [None, 784])

# 应变量的10个结果 形如[0 0 0 0 0 0 1 0 0 0]
# 则这种情况下  该图片为”6“
y = tf.placeholder("float", [None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#**LOGISTIC REGRESSION MODEL
# 这10列里，每1列记录了 =0~9 相似程度一个分值
actv = tf.nn.softmax(tf.matmul(x, w)+b)

# Cost Fonction
# 损失值 = -log（p） 其中p为 分值 归一化之后的概率
# 平均的LOST
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))
# Optimizer
# 梯度下降法 做优化 学习率为0.01, 最小化cost
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

                                                      
# Prediction
# 预测值索引 与 label索引是否一致
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))

# Accuracy
# 求出正确率
accr = tf.reduce_mean(tf.cast(pred, "float"))

# Initializer
init = tf.global_variables_initializer()

# tf.argmax(arr， 0).eval() // 0→按列求出矩阵最大值索引，
# tf.argmax(arr， 1).eval() // 1→按行求出矩阵最大值索引，
sess = tf.InteractiveSession()


                                                      
#############################

# 全局样本迭代次数
training_epochs = 100
                
# 每次迭代选用样本数量                                     
batch_size = 100     

# 展示
display_step = 5                                                 

#session
sess = tf.Session()
sess.run(init)

# Mini-Batch Learning
for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int (mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            #在100个样本中一次次进行返回
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #调用了梯度下降求解  用dict形式  给placehold回传
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            #
            feeds = {x: batch_xs, y: batch_ys}
            #累加计算平均损失值
            avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
            
        #display
        #每5个epoch打印一次display
        if epoch % display_step == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: mnist.test.images , y: mnist.test.labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print ("Epoch: %03d %03d cost: %.9f train_acc: %.3f test_acc: %.3f"  % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print ("DONE")
            






                                                
#for i in randidx:
#    curr_img = np.reshape(trainimg[i, :], (28, 28)) # 28 28 matrix
    




#%%