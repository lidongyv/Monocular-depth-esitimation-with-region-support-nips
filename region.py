# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-26 10:28:48
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-26 18:43:56
import os
import numpy as np
import cv2
import copy
def seg2regions(seg):
    region=seg
    label=-1
    while(np.max(seg)>0):
        category=np.max(seg)
        while(np.max(seg)==category):
            positions=np.array(np.where(seg==category))

            region[positions[0,0],positions[1,0]]=label
            for i in range(1,positions.shape[1]):
                if np.abs(positions[0,i]-positions[0,i-1])+np.abs(positions[1,i]-positions[1,i-1])<=2:
                     region[positions[0,i],positions[1,i]]=label
                elif region[positions[0,i],positions[1,i]-1]==label or\
                        region[positions[0,i]-1,positions[1,i]+1]==label or\
                        region[positions[0,i]-1,positions[1,i]]==label or\
                        region[positions[0,i]-1,positions[1,i]-1]==label:
                        region[positions[0,i],positions[1,i]]=label
                else:
                    break;
        label=label-1
    return np.abs(region)

def reg2supports(reg,dep,variance):
    supports=np.zeros([reg.shape[0],reg.shape[1]])
    for i in range(1,int(np.max(reg)+1)):
        region=np.where(reg==i)
        depth=dep[region]
        if(np.max(depth)-np.min(depth)<2*variance):
            supports[region]=np.mean(depth)

        else:
            start=np.min(depth)
            end=np.min(depth)+variance
            while(end<np.max(depth)):
                region1=np.where(np.logical_and(reg==i,np.logical_and(dep>start,dep<end)))
                supports[region1]=np.mean(dep[region1])
                if end+variance<np.max(depth):
                    start=end
                    end=end+variance
                else:
                    region1=np.where(np.logical_and(reg==i,np.logical_and(dep>start,dep<np.max(depth))))
                    supports[region1]=np.mean(dep[region1])
                    break;
    return supports                    

filenames=os.listdir('/home/lidong/Documents/datasets/nyu/tmp')
for i in range(len(filenames)):
    data=np.load(os.path.join('/home/lidong/Documents/datasets/nyu/tmp',filenames[i]))
    img = data[:,:,0:3]
    depth = data[:,:,3]
    segments = data[:,:,4]    
    regions=seg2regions(copy.deepcopy(segments))
    variance=(np.max(depth)-np.min(depth))/10
    supports=reg2supports(copy.deepcopy(regions),copy.deepcopy(depth),variance)
    supportv=seg2regions(copy.deepcopy(supports))
    # data=np.concatenate(data,np.reshape(regions,[regions.shape[0],regions.shape[1],1]),2)
    # data=np.concatenate(data,np.reshape(supports,[supports.shape[0],supports.shape[1],1]),2)
    # print(np.max(regions),np.max(regions),np.max(regions))
    a=np.reshape(regions/np.max(regions),[supports.shape[0],supports.shape[1],1])
    b=np.reshape(supports/np.max(supports),[supports.shape[0],supports.shape[1],1])
    region_support=np.concatenate((a,b),2)
    c=np.reshape(supportv/np.max(supportv),[supportv.shape[0],supportv.shape[1],1])
    region_support=np.concatenate((region_support,c),2)
    cv2.namedWindow('show')
    cv2.imshow('show',region_support)
    cv2.namedWindow('show2')
    cv2.imshow('show2',b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
