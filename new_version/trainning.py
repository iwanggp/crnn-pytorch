#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :trainning.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/10/6
@Desc   : 训练的主代码
=================================================='''
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn import CTCLoss
import os
import utils
from data import dataset
import models.crnn as net
import config
from pathlib import Path

expr_dir = Path(config.expr_dir)
if not expr_dir.exists():  # 如果不存在该文件夹就创建
    expr_dir.mkdir(parents=True)
# ------------------------------------------------
"""
加载训练数据和验证数据集
"""


def data_loader():
    train_dataset = dataset.CRNNDataset(root=config.TRAIN_ROOT, transform=config.transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchSize,
                                               shuffle=True, num_workers=int(config.workers))
    val_dataset = dataset.CRNNDataset(root=config.VAL_ROOT,
                                      transform=dataset.ResizeNormalize(
                                          (config.imgW, config.imgH)))  # 验证集一定要进行transfrom处理
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=config.batchSize,
                                             num_workers=int(config.workers))

    return train_loader, val_loader


"""
1 网络初始化
2 权重初始化
3 是否加载预训练权重
"""


def weights_init(m):
    """
    权重初始化
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


"""
网络的初始化
"""


def net_init():
    nclass = len(config.alphabet) + 1  # 一定要加一个空格
    crnn = net.CRNN(config.imgH, config.nc, nclass, config.nh)
    crnn.apply(weights_init)
    if config.pretrained != '':  # 预训练从中断训练
        print('loading pertrained model from %s' % config.pretrained)
        if config.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(config.pretrained))
    return crnn


# ------------------------------验证集验证结果--------
def val(crnn, criterion, val_loader, device, converter):
    print('Start val')
    crnn.eval()
    n_correct = 0
    loss_avg = utils.averager()  # The blobal loss_avg is used by train
    for i, data in enumerate(val_loader):
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        image = cpu_images.to(device)
        text, length = converter.encode(cpu_texts)
        text = text.to(device)
        length = length.to(device)
        with torch.no_grad():
            preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = cpu_texts
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / len(val_loader.dataset)  # float(len(val_loader) * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return accuracy


# ---------------------------------训练的主代码--------
def train_batch(crnn, criterion, optimizer, data, device, convert):
    crnn.train()
    cpu_images, cpu_texts = data
    cpu_images = cpu_images.to(device)
    batch_size = cpu_images.size(0)
    text, length = convert.encode(cpu_texts)
    text = text.to(device)
    length = length.to(device)
    image = cpu_images
    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = torch.LongTensor([preds.size(0)] * batch_size)
    cost = criterion(preds, text, preds_size, length) / batch_size
    cost.backward()
    optimizer.step()
    return cost


# -----------------------训练主代码主入口-------------
def train():
    train_loader, val_loader = data_loader()  # 加载训练集和验证集
    crnn = net_init()
    print(crnn)
    converter = utils.strLabelConverter(config.alphabet)  # 加载label解码
    if config.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    elif config.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=config.lr)

    criterion = CTCLoss(zero_infinity=True)
    crnn = crnn.to(config.device)
    acc = 0
    curAcc = 0
    for epoch in range(config.nepoch):
        n = len(train_loader)
        interval = n // 2  # 评估模型
        pbar = utils.Progbar(target=n)
        loss = 0
        for i, data in enumerate(train_loader):
            cost = train_batch(crnn, criterion, optimizer, data, config.device, converter)
            loss += cost.data.cpu().numpy()
            if (i + 1) % interval == 0:
                curAcc = val(crnn, criterion, val_loader, config.device, converter)
                if curAcc > acc:
                    checkpoint = {
                        'model': crnn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': i + 1
                    }
                    acc = curAcc
                    torch.save(checkpoint, f'{expr_dir}/bestAcc.pth')
                    torch.save(crnn.state_dict(), 'crnn.pth')  # 只保存模型的结构
            pbar.update(i + 1, values=[('loss', loss / ((i + 1) * config.batchSize)), ('acc', curAcc)])
            print(loss / ((i + 1) * config.batchSize))


if __name__ == '__main__':
    train()
