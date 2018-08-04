# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-07-31 20:35:41
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-03 22:00:32
import os
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import time
import cv2

COLOR=[np.array([255,0,0]), 
       np.array([0,255,0]),
       np.array([0,0,255]),
       np.array([125,125,0]),
       np.array([0,125,125]),
       np.array([125,0,125]),
       np.array([50,100,50]),
       np.array([100,50,100])
       ]

def cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth, bin_seeding=True)
    print ('Mean shift clustering, might take some time ...')
    tic = time.time()
    ms.fit(prediction)
    print ('time for clustering', time.time() - tic)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    num_clusters = cluster_centers.shape[0]

    return num_clusters, labels, cluster_centers

def get_instance_masks(prediction, bandwidth):
    batch_size, h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):
        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([h*w, feature_dim]), bandwidth)
        print ('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h,w])
        mask = np.zeros([h,w], dtype=np.uint8)

        num_clusters = min([num_clusters,8])
        for mask_id in range(num_clusters):
            ind = np.where(labels==mask_id)
            mask[ind] = np.uint8(255/num_clusters*mask_id)


        instance_masks.append(mask)

    return np.array(instance_masks)


def save_instance_masks(prediction,output_dir, bandwidth, count):
    batch_size, h, w, feature_dim = prediction.shape

    instance_masks = []
    for i in range(batch_size):
        num_clusters, labels, cluster_centers = cluster(prediction[i].reshape([h*w, feature_dim]), bandwidth)
        print ('Number of predicted clusters', num_clusters)
        labels = np.array(labels, dtype=np.uint8).reshape([h,w])
        mask = np.zeros([h,w,3], dtype=np.uint8)

        num_clusters = min([num_clusters,8])
        for mask_id in range(num_clusters):
            mask = np.zeros([h,w,3], dtype=np.uint8)
            ind = np.where(labels==mask_id)
            mask[ind] = np.array([255,255,255])
            output_file_name = os.path.join(output_dir, 'cluster_{}_{}.png'.format(str(count).zfill(4), str(mask_id)))
            cv2.imwrite(output_file_name, mask)


        instance_masks.append(mask)

    return instance_masks
def mean_shift(feature,mean,bandwidth):
    #feature shape c h w
    dis=feature-mean
    dis=torch.norm(dis,dim=0)
    mask=torch.where(dis<bandwidth,torch.tensor(1),torch.tensor(0)).float()
    mean=torch.sum((feature*mask).view(feature.shape[0],feature.shape[1]*feature.shape[2]),dim=1)/torch.sum(mask)
    return mean
def get_mask(feature,mean,bandwidth):
    dis=feature-mean
    dis=torch.norm(dis,dim=0)
    mask=torch.where(dis<bandwidth,torch.tensor(1),torch.tensor(0).float())
    pixels=mask.nonzero()
    minx=torch.min(pixels[0,:])
    maxx=torch.max(pixels[0,:])
    miny=torch.min(pixels[1,:])
    maxy=torch.max(pixels[1,:])
    areas=torch.ceil((maxx-minx)/60)*torch.ceil((maxy-miny)/60)
    small=[]
    for i in range(1,torch.ceil((maxx-minx)/60).int()+1):
        for j in range(1,torch.ceil((maxy-miny)/60)+1):
            mask[minx+60*(i-1):minx+60*i,miny+60*(j-1):miny+60*j]*=i*j
    areas=torch.unique(mask).sort()[0]
    for i in range(1,len(unique)+1):
        torch.where(mask==areas[i],-i,mask)
    mask=-mask
    areas=len(areas)-1
    return mask,areas

def re_label(mask,area,bandwidth):
    index=torch.sum(area)
    count=0
    for i in range(area.shape[0]):
        mask[i,:,:]=torch.where(mask[i,:,:]>0,mask[i,:,:]+count,mask[i,:,:])
        count+=area[i]
    segment=torch.where(mask>0,torch.tensor(1),torch.tensor(0).float())
    final=torch.sum(mask,dim=0)/torch.sum(segment,dim=0)
    return mask,count,segment
def fast_cluster(feature,label,bandwidth=0.7):
    masks=[]
    areas=[]
    for i in range(feature.shape[0]):
        n_feature=feature[i,...]
        label=torch.zeros(n_feature.shape[1],n_feature.shape[2])
        n_masks=[]
        n_areas=[]
        while(torch.min(labels)==0):
            candidate=torch.where(labels==0,torch.tensor(1).float().cuda(),torch.tensor(0).float().cuda()).nonzero()
            seed=torch.randint(len(candidate),(1,))[0]
            mean=feature[:,candidate[seed][0],candidate[seed][1]].view(n_feature.shape[0],1,1)
            mean=mean_shift(n_feature, mean, bandwidth)
            n_masks,n_areas=get_mask(feature, mean, bandwidth)
            n_masks.append(n_masks)
            n_areas.append(n_areas)
            label+=n_masks
        masks.append(torch.stack(n_masks))
        areas.append(torch.stack(n_areas))
    masks=torch.stack(masks)
    areas=torch.stack(areas)
    mask,count,regions=relable(masks,areas)
    return mask,count,regions
    #x=torch.arange(0,feature.shape[2])
    #y=torch.arange(0,feature.shape[3])