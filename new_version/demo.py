#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :demo.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/10/6
@Desc   :
=================================================='''
import os
import shutil
import time
import cv2
import torch
import torchvision
from torch.autograd import Variable
import utils
import models.crnn as crnn
import config
from data import dataset

model_path = f'expr/{config.NAME}_498.pth'
image_dir = r'./ones'
mistake_dir = './train_mistake'
os.makedirs(mistake_dir, exist_ok=True)
device = 'cpu'

nclass = len(config.alphabet) + 1
model = crnn.CRNN(config.imgH, config.nc, nclass, config.nh)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
converter = utils.strLabelConverter(config.alphabet)

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度
    scale = image.shape[0] * 1.0 / 32
    w = image.shape[1] / scale
    w = int(w)
    print(f"scale........w is...........{w}")
    transformer = dataset.CV2ResizeNormalize((w, config.imgH))
    image = transformer(image)
    torchvision.utils.save_image(image, "./train_mistake/" + image_name)
    image = image.to(device)
    image = image.view(1, *image.size())
    tic = time.time()
    with torch.no_grad():
        preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    toc = time.time()
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[-1].split('-')[0]
    print(f'{image_name == sim_pred} %-20s => %-20s   |   {image_name}' % (raw_pred, sim_pred))
    if image_name != sim_pred:
        shutil.copy(image_path, os.path.join(mistake_dir, f'{sim_pred}---{image_name}.jpg'))
