# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-04-25 23:06:40
# @Last Modified by:   yulidong
# @Last Modified time: 2018-07-25 22:42:58


import os
import torch
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from python_pfm import *
from rsden.utils import recursive_glob
import torchvision.transforms as transforms
from PIL import Image

class KITTI(data.Dataset):


    def __init__(self, root, split="train", img_size=(375,1242),is_transform=True,task='all'):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.shape=img_size
        self.root = root
        self.split = split
        self.num=0
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.path=os.path.join(self.root,self.split)
        self.images=np.load(os.path.join(self.path,'kitti_images.npy'))
        self.grounds=np.load(os.path.join(self.path,'kitti_ground.npy'))
        if len(self.images)<1:
            raise Exception("No files for %s found in %s" % (split, self.path))

        print("Found %d in %s images" % (len(self.images), self.path))
    def __len__(self):
        """__len__"""
        return len(self.images)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        #data=np.load(os.path.join(self.path,self.files[index]))
        img = Image.open(self.images[index])
        #dis=np.array(dis[0], dtype=np.uint8)

        depth = Image.open(self.grounds[index])
        if self.is_transform:
            img, depth= self.transform(img, depth)

        return img, depth
    def get_params(self,img):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th=360
        tw=800
        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw
    def transform(self, img, depth):
        """transform

        :param img:
        :param depth:
        """
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        #img = torch.from_numpy(img).float()

        #img = img.astype(float) / 255.0
        # NHWC -> NCHW
        #img = img.transpose(1,2,0)
        #img=np.array(img)
        totensor=transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        #padding=transforms.Pad(padding=(0,0,0,0),padding_mode='edge')
        i,j,h,w=self.get_params(img)
        #img=padding(img)
        depth=depth.crop((j, i, j + w, i + h)) 
        depth=np.array(depth).astype(np.float32)/256
        depth=torch.from_numpy(depth)
        img=img.crop((j, i, j + w, i + h)) 
        img=totensor(img)
        img=normalize(img)      
        #depth=padding(depth)

        #print(img.shape,depth.shape)
        return img,depth
