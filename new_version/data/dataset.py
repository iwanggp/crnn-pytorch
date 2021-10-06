#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :dataset.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/10/6
@Desc   :数据集处理的工具类
=================================================='''
import os
from os.path import join
from os.path import split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from PIL import Image
from deep_utils import split_extension
import config


class CRNNDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.image_paths = [join(root, img_name) for img_name in os.listdir(root) if
                            split_extension(img_name)[-1].lower() in ['.jpg', '.png', '.jpeg']]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.image_paths[index]
        if config.nc == 1:
            img = Image.open(img_path).convert('L')
        else:
            img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        label = self.get_label(img_path)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    @staticmethod
    def get_label(img_path):
        label = split_extension(split(img_path)[-1])[0]
        label = label.split('-')[0]
        return label


# 归一化
class ResizeNormalize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)  # 图片的处理，以后转换用mean 127.5来还原
        return img


# Opencv的处理方式来处理
class CV2ResizeNormalize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


# 每一个批次的图片处理流程
class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # if self.keep_ratio:
        #     ratios = []
        #     for image in images:
        #         w, h = image.size
        #         ratios.append(w / float(h))
        #     ratios.sort()
        #     max_ratio = ratios[-1]
        #     imgW = int(np.floor(max_ratio * imgH))
        #     imgH = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        transform = ResizeNormalize(imgW, imgH)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
