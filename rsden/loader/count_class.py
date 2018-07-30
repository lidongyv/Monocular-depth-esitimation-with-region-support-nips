# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-28 09:12:33
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-28 18:44:01
import numpy as np
import os
file_dir='/home/lidong/Documents/datasets/nyu/nyu2/train/'
files_name=os.listdir(file_dir)
files_name.sort(key=lambda x:int(x[:-4]))
max=0
for i in range(len(files_name)):
    data=np.load(os.path.join(file_dir,files_name[i]))
    region_instance=data[...,7]
    if np.max(region_instance)>=max:
        max=np.max(region_instance)
        print(max)
#max=64