#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :config.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/10/6
@Desc   :配置文件中配置信息
=================================================='''
import torch
from deep_utils import AugmentTorch

# about data and net

# TRAIN_ROOT, VAL_ROOT = './images', \
#                        r'./images'
from utils import get_alphabet

TRAIN_ROOT, VAL_ROOT = r'D:\chromdownload\text2image-master\text2image-master\results', \
                       r'D:\chromdownload\text2image-master\text2image-master\val'
NAME = 'crnn_rec'
keep_ratio = False  # whether to keep ratio for image resize
manualSeed = 1234  # reproduce experiemnt
random_sample = True  # whether to sample the hand_labeled_dataset with random sampler
imgH = 32  # the height of the input image to network
imgW = 280  # the width of the input image to network
nh = 256  # size of the lstm hidden state
nc = 1
pretrained = ''  # path to pretrained model (to continue training)
expr_dir = 'expr'  # 保存权重
dealwith_lossnan = False  # whether to replace all nan/inf in gradients to zero

# hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # enables cuda
multi_gpu = False  # whether to use multi gpu
ngpu = 1  # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0  # number of data loading workers

# training process
displayInterval = 100  # interval to be print the train loss
valInterval = 1000  # interval to val the model loss and accuracy
saveInterval = 1000  # interval to save model
n_val_disp = 10  # number of samples to display when val the model
alphabet = get_alphabet('keys.txt')
# finetune
nepoch = 5000  # number of epochs to train for
batchSize = 64  # input batch size
lr = 0.0001  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
adam = True  # whether to use adam (default is rmsprop)
adadelta = False  # whether to use adadelta (default is rmsprop)
if nc == 1:
    mean, std = [0.5], [0.5]
else:
    mean, std = [0.5] * nc, [0, 5] * nc

transformations = AugmentTorch.get_augments(
    AugmentTorch.resize((imgH, imgW)),
    AugmentTorch.normalize(mean=mean, std=std),
    AugmentTorch.random_rotation((-20, 20)),
    AugmentTorch.gaussian_blur(kernel_size=3),

)
