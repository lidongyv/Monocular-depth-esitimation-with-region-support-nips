# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-05 19:13:11
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-05 19:13:30
import scipy.io as sio    
import matplotlib.pyplot as plt    
import numpy as np
import h5py    
data =  h5py.File('/home/lidong/Documents/datasets/nyu/nyu_depth_data_labeled.mat')
keys=[]
values=[]
#shapes=[]
for k, v in data.items():
    keys.append(k)
    values.append(v)
    print(v)

depths=data['depths']
images=data['images']
datas=[]
for i in range(2284):
    image=np.array(images[i,:,:,:].astype('float32'))
    depth=np.array(depths[i,:,:])
    depth=np.reshape(depth,[1,depth.shape[0],depth.shape[1]])
    group=np.concatenate((image,depth),0)
    group=np.transpose(group,[0,2,1])
    datas.append(group)
np.save('/home/lidong/Documents/datasets/nyu/nyu_depth_data_labeled.npy',datas)