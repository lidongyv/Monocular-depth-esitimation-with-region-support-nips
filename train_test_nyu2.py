# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 19:03:52
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-27 22:50:27
import scipy.io
import numpy as np
import os
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/nyu2_test_index.mat')
test1=data['testNdxs']
for i in range(len(test1)):
    filename=str(test1[i][0]-1)+'.npy'
    print(os.path.join('home/lidong/Documents/datasets/nyu/nyu2/train/',filename))
    os.rename(os.path.join('/home/lidong/Documents/datasets/nyu/nyu2/train/',filename), \
        os.path.join('/home/lidong/Documents/datasets/nyu/nyu2/test/',filename))