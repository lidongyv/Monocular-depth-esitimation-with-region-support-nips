# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 16:31:14
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-04 10:23:15

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    #print(c,target.max().data.cpu().numpy())

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    #loss=log_p.sum()
    loss = F.nll_loss(log_p, target,
                      weight=weight, size_average=False)
    #print(loss, mask.data.sum())
    if size_average:
    #    print(mask.data.sum())
       loss /= mask.data.sum()
    #    loss=loss/(950*540)
    return loss
def l1(input, target, weight=None, size_average=True):
    target=torch.reshape(target,(input.shape))
    #print(input.shape)
    #print(target.shape)
    # num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    # positive=num/torch.sum(torch.ones_like(input))
    #print(positive.item())
    loss=nn.MSELoss()
    relation=torch.sqrt(loss(input,target))
    #mean=torch.abs(torch.mean(input)-torch.mean(target))
    #print("pre_depth:%.4f,ground_depth:%.4f"%(torch.mean(input[1]).data.cpu().numpy().astype('float32'),torch.mean(target).data.cpu().numpy().astype('float32')))
    #output=relation+0.2*mean
    return relation
def log_loss(input, target, weight=None, size_average=True):
    # num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    # positive=num/torch.sum(torch.ones_like(input))
    # print(positive.item())
    target=torch.reshape(target,(input.shape))
    loss=nn.MSELoss() 
    input=torch.log(input+1e-12) 
    target=torch.log(target+1e-12) 
    #relation=torch.sqrt(loss(input,target)) 
    relation=loss(input,target) 
    d=0.5*torch.pow(torch.sum(input-target),2)/torch.pow(torch.sum(torch.ones_like(input)),2)
 
    return relation-d 

    # target=torch.reshape(target,(input.shape))
    # #loss=nn.MSELoss()
    # num=torch.sum(torch.where(input>0,torch.ones_like(input),torch.zeros_like(input)))
    # input=torch.log(torch.where(input>0,input,torch.ones_like(input)))
    # target=torch.log(torch.where(target>0,target,torch.ones_like(target)))
    # # #relation=torch.sqrt(loss(input,target))
    # relation=torch.sum(torch.pow(torch.where(input==0,input,input-target),2))/num
    # d=torch.pow(torch.sum(torch.where(input==0,input,input-target)),2)/torch.pow(num,2)*0.5
    # #positive=num/torch.sum(torch.ones_like(input))
    # #print(positive.item())
    # #-torch.sum(torch.where(input<0,input,torch.zeros_like(input)))/num
    # losses=relation+d
    # return losses

def log_l1(input, target, weight=None, size_average=True):
    l1loss=l1(input,target)
    logloss=log_loss(input,target)
    num=torch.sum(torch.where(input==0,torch.ones_like(input),torch.zeros_like(input)))
    positive=num/torch.sum(torch.ones_like(input))
    print(positive.item())
    loss=(1-positive)*logloss+positive*l1loss
    return loss


def region(pre,supportd,supporti):
    loss=torch.zeros(1)
    for i in range(1,torch.max(supporti)):
        pre_region=torch.where(supporti==i,pre,torch.zeros_like(pre))
        ground_region=torch.where(supporti==i,supportd,torch.zeros_like(pre))
        num=torch.sum(torch.where(supporti==i,torch.ones_like(pre),torch.zeros_like(pre)))
        loss+=0.3*torch.abs(pre_region-ground_region)/num
        average=torch.sum(pre_region)/num
        variance=torch.sum(torch.where(pre_region>0,torch.pow(pre_region-average,2),0))/num
        loss+=0.7*variance



