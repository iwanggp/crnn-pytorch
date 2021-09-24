#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :trainning.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/9/24
@Desc   :训练主脚本
=================================================='''
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from torch.nn import CTCLoss

from release.crnn import CRNN
from release.datasets import PathDataset, randomSequentialSampler, alignCollate, resizeNormalize
from release.utils import strLabelConverter, loadData, Progbar

roots = glob('E:/datasets/crnn/images/*.jpg')
alphabetChinese = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
trainP, testP = train_test_split(roots, test_size=0.1)  ##此处未考虑字符平衡划分
traindataset = PathDataset(trainP, alphabetChinese)
testdataset = PathDataset(testP, alphabetChinese)

batchSize = 32
workers = 0
imgH = 32
imgW = 280
keep_ratio = True
cuda = True
ngpu = 1
nh = 256
sampler = randomSequentialSampler(traindataset, batchSize)
train_loader = torch.utils.data.DataLoader(
    traindataset, batch_size=batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

model = CRNN(32, 1, len(alphabetChinese) + 1, 256, True)
lr = 0.0001
optimizer = optim.Adadelta(model.parameters(), lr=lr)
converter = strLabelConverter(''.join(alphabetChinese))
criterion = CTCLoss()
image = torch.FloatTensor(batchSize, 3, imgH, imgW)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])  ##转换为多GPU训练模型
    image = image.cuda()
    criterion = criterion.cuda()


#
def trainBatch(net, criterion, optimizer, cpu_images, cpu_texts):
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    loadData(text, t)
    loadData(length, l)
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def predict(im):
    """
    图片预测
    """
    image = im.convert('L')
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def val(net, dataset, max_iter=100):
    # for p in net.parameters():
    #     p.requires_grad = False
    net.eval()
    i = 0
    n_correct = 0
    N = len(dataset)

    max_iter = min(max_iter, N)
    for i in range(max_iter):
        im, label = dataset[np.random.randint(0, N)]
        if im.size[0] > 1024:
            continue

        pred = predict(im)
        if pred.strip() == label:
            n_correct += 1

    accuracy = n_correct / float(max_iter)
    return accuracy


acc = 0
for i in range(1000):

    print('epoch:{}/{}'.format(i, 100))
    n = len(train_loader)
    pbar = Progbar(target=n)
    interval = n // 2  ##评估模型
    train_iter = iter(train_loader)
    loss = 0
    for j in range(n):
        model.train()
        cpu_images, cpu_texts = train_iter.next()
        cost = trainBatch(model, criterion, optimizer, cpu_images, cpu_texts)

        loss += cost.data.cpu().numpy()

        if (j + 1) % interval == 0:
            curAcc = val(model, testdataset, max_iter=1024)
            if curAcc > acc:
                acc = curAcc
                torch.save(model.state_dict(), 'modellstm.pth')
        pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])
        print(loss / ((j + 1) * batchSize))
