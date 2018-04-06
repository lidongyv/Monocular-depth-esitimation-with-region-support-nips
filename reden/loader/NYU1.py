# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-05 16:40:02
# @Last Modified by:   yulidong
# @Last Modified time: 2018-04-06 20:49:32

import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
from rsnet.utils import recursive_glob


class NYU1(data.Dataset):


    def __init__(self, root, split="train", is_transform=True, img_size=(480,640)):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.num=0
        self.is_transform = is_transform
        self.n_classes = 9  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (480, 640)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.data=np.load(root+split+'.npy')
        if not self.data:
            raise Exception("No files for ld=[%s] found in %s" % (split, self.root))

        print("Found %d in %s images" % (len(self.data), split))

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """

        img = self.data(index,0:2,:,:)
        #dis=readPFM(disparity_path)
        #dis=np.array(dis[0], dtype=np.uint8)

        region = self.data(index,3,:,:)

        if self.is_transform:
            img, region = self.transform(img, region)

        return img, region

    def transform(self, img, region):
        """transform

        :param img:
        :param region:
        """
        img = img[:, :,:]
        img = img.astype(np.float64)
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #img = img.transpose(2, 0, 1)

        #region=region[0,:,:]
        #region = region.astype(float)/32
        #region = np.round(region)
        #region = m.imresize(region, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        #region = region.astype(int)
        #region=np.reshape(region,[1,region.shape[0],region.shape[1]])
        #classes = np.unique(region)
        #print(classes)
        #region = region.transpose(2,0,1)
        #if not np.all(classes == np.unique(region)):
        #    print("WARN: resizing labels yielded fewer classes")

        #if not np.all(classes < self.n_classes):
        #    raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        region = torch.from_numpy(region).long()

        return img, region
