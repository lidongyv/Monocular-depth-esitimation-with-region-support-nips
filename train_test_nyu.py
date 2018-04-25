# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 19:03:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-25 22:59:12
import scipy.io
import numpy as np
import os
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/test_files.mat')
test1=data['test1']
test2=data['test2']
for i in range(len(test1)):
    filename=str(test1[i][0])+'.npy'
    #print(os.path.join('home/lidong/Documents/datasets/nyu/train/',filename))
    os.rename(os.path.join('/home/lidong/Documents/datasets/nyu/train/',filename),os.path.join('/home/lidong/Documents/datasets/nyu/test/',filename))
for i in range(len(test2)):
    filename=str(test2[i][0])+'.npy'
    #print(filename)
    os.rename(os.path.join('/home/lidong/Documents/datasets/nyu/train/',filename),os.path.join('/home/lidong/Documents/datasets/nyu/hard_test/',filename))
# import matplotlib.pyplot as plt    
# import numpy as np
# import h5py
# import os    
# data =  h5py.File('/home/lidong/Documents/datasets/nyu/nyu_depth_v2_labeled.mat')
# names=data['scenes']
# print(str(names[0,0]))

