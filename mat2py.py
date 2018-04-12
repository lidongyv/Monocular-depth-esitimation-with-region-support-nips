# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-12 12:34:57
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-12 14:33:48
import scipy.io
import numpy as np
data=scipy.io.loadmat('/home/lidong/Documents/datasets/nyu/train_test.mat')
print('dasf')
train_depth=data['train_depth']
print(train_depth.shape)
train_image=data['train_image']
print(train_image.shape)
test_depth=data['test_depth']
test_image=data['test_image']
num=train_depth.shape[-1]
train_depth=np.reshape(train_depth,[480,640,1,num])
num=test_depth.shape[-1]
test_depth=np.reshape(test_depth,[480,640,1,num])
train=np.concatenate((train_image,train_depth),2)
test=np.concatenate((test_image,test_depth),2)
np.save('/home/lidong/Documents/datasets/nyu/train_split.npy',train)
np.save('/home/lidong/Documents/datasets/nyu/test_split.npy',test)
print('done')