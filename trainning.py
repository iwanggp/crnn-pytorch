#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :trainning.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/9/24
@Desc   :训练主脚本
=================================================='''
import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn import CTCLoss

from crnn import CRNN
from datasets import alignCollate, randomSequentialSampler, resizeNormalize, PathDataset
from utils import strLabelConverter, loadData, Progbar

from data.words import Word
from generator import Generator

word = Word()
# 单次训练
def trainBatch(net, criterion, optimizer, cpu_images, cpu_texts, image, converter, text, length):
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


def predict(im, model, converter):
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


def val(net, converter, dataset, max_iter=100):
    '''
    图片验证程序
    :param net:
    :param converter:
    :param dataset:
    :param max_iter:
    :return:
    '''
    net.eval()
    n_correct = 0
    N = len(dataset)

    max_iter = min(max_iter, N)
    for i in range(max_iter):
        im, label = dataset[np.random.randint(0, N)]
        if im.size[0] > 1024:
            continue

        pred = predict(im, net, converter)
        if pred.strip() == label:
            n_correct += 1

    accuracy = n_correct / float(max_iter)
    return accuracy


def train(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    imgs_labels = glob(f'{args.image_dir}/*.jpg')
    with open(args.alphabetChinese, 'r') as f:
        alphabetChinese = ''.join(f.readlines())  # 获得编码表
    trainP, testP = train_test_split(imgs_labels, test_size=0.1)  ##此处未考虑字符平衡划分
    traindataset = PathDataset(trainP, alphabetChinese)
    testdataset = PathDataset(testP, alphabetChinese)
    batchSize = args.batch_size
    workers = args.workers
    imgH = args.imgH
    imgW = args.imgW
    keep_ratio = True
    lr = args.lr
    nh = args.hidden
    channel = args.channel
    model = CRNN(imgH, channel, len(alphabetChinese) + 1, nh, True)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    converter = strLabelConverter(''.join(alphabetChinese))
    criterion = CTCLoss()
    image = torch.FloatTensor(batchSize, channel, imgH, imgW)
    text = torch.IntTensor(batchSize * 5)
    length = torch.IntTensor(batchSize)
    sampler = randomSequentialSampler(traindataset, batchSize)
    epochs = args.epochs
    init_epoch = 0
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(workers),
        collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])  ##转换为多GPU训练模型
        image = image.cuda()
        criterion = criterion.cuda()
    if args.resume:  # 是否接着训练
        # if not os.path.exists(f'{args.out_dir}/bestAcc.pt'):
        #     return
        assert os.path.exists(f'{args.output_dir}/bestAcc.pt'), 'do you have bestAcc.pt'
        checkpoint = torch.load(
            os.path.join(args.output_dir, 'bestAcc.pt'.format(args.direction, args.init_epoch)),
            map_location='cpu')
        init_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
    for i in range(init_epoch, args.epochs):
        print('epoch:{}/{}'.format(i, epochs))
        n = len(train_loader)
        pbar = Progbar(target=n)
        interval = n // 2  ##评估模型
        train_iter = iter(train_loader)
        loss = 0
        acc = 0

        for j in range(n):
            model.train()
            cpu_images, cpu_texts = train_iter.next()
            cost = trainBatch(model, criterion, optimizer, cpu_images, cpu_texts, image, converter, text, length)

            loss += cost.data.cpu().numpy()

            if (j + 1) % interval == 0:
                curAcc = val(model, converter, testdataset, max_iter=1024)
                if curAcc > acc and args.output_dir:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': i + 1,
                        'args': args
                    }
                    acc = curAcc
                    torch.save(checkpoint, f'{args.output_dir}/bestAcc.pt')
                    torch.save(model.state_dict(), 'best.pt')  # 只保存模型的结构
            pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])
            print(loss / ((j + 1) * batchSize))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help='cpu or cuda')
    parser.add_argument("--image_dir", type=str, default='/Users/gongpengwang/Documents/advance/CRNN/images',
                        help='the images dir')
    parser.add_argument("--direction", type=str, choices=['horizontal', 'vertical'], default='horizontal',
                        help='horizontal or vertical')
    parser.add_argument("--batch_size", type=int, default=32, help='batch_size')
    parser.add_argument("--resume", type=bool, default=False, help='batch_size')
    parser.add_argument("--imgH", type=int, default=32, help='img height')
    parser.add_argument("--imgW", type=int, default=280, help='img width')
    parser.add_argument("--channel", type=int, default=1, help='channel of image')
    parser.add_argument("--hidden", type=int, default=256, help='hidden ')
    parser.add_argument("--alphabetChinese", type=str, default='keys.txt', help='the alphabet file')
    parser.add_argument("--epochs", type=int, default=90, help='epochs')
    parser.add_argument("--init-epoch", type=int, default=0, help='init epoch')
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay (default: 0)')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    parser.add_argument("--workers", type=int, default=0, help="number of workers")
    parser.add_argument('--output_dir', default='./output', help='path where to save')
    # 训练的主代码
    arguments = parser.parse_args()
    train(arguments)
