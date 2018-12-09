# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:39:43 2018

@author: TilkeyYANG
"""
 
# Inputting with OpenCV, Channel order is BGR
# matplotlib Channel order is RGB

import cv2
from matplotlib import pyplot as plt

import os
os.chdir('.')
cwd = os.getcwd() 

# Inputting an photo
img = cv2.imread('./tests/Test1.jpg')

# Splitting channels
B, G, R = cv2.split(img)

# Converting
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

plt.figure('BGR_RGB')
 
# Affiche the input img with BGR Channels
plt.subplot(2,2,1), plt.imshow(img)
# B Channel
plt.subplot(2,2,2), plt.imshow(B)
# G Channel
plt.subplot(2,2,3), plt.imshow(G)
# R Channel
plt.subplot(2,2,4), plt.imshow(R)
 
plt.show()