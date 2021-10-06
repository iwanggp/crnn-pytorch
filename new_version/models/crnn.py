#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :crnn.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/10/6
@Desc   :crnn模型主结构
=================================================='''
import torch.nn as nn
import torch.nn.functional as F
from deep_utils import BlocksTorch


# 双向BLSTM结构
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T*b,nOut]
        output = output.view(T, b, -1)
        return output


# CRNN的主体结构
class CRNN(nn.Module):
    """
    CRNN的结构
    """

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        activation = 'relu'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        if leakyRelu:
            activation = 'leakyrelu'
        block_0 = BlocksTorch.conv_norm_act(nc, nm[0], k=ks[0], s=ss[0], p=ps[0], act=activation, index=0,
                                            pooling='max', norm=None)
        block_1 = BlocksTorch.conv_norm_act(nm[0], nm[1], k=ks[1], s=ss[1], p=ps[1], act=activation, index=1,
                                            pooling='max', norm=None)
        block_2 = BlocksTorch.conv_norm_act(nm[1], nm[2], k=ks[2], s=ss[2], p=ps[2], act=activation, index=2,
                                            pooling=None, norm='bn')
        block_3 = BlocksTorch.conv_norm_act(nm[2], nm[3], k=ks[3], s=ss[3], p=ps[3], act=activation, index=3,
                                            pooling='max',
                                            pooling_s=(2, 1), pool_kwargs=dict(padding=(0, 1)), norm=None)
        block_4 = BlocksTorch.conv_norm_act(nm[3], nm[4], k=ks[4], s=ss[4], p=ps[4], act=activation, index=4,
                                            pooling=None, norm='bn')
        block_5 = BlocksTorch.conv_norm_act(nm[4], nm[5], k=ks[5], s=ss[5], p=ps[5], act=activation, index=5,
                                            pooling='max',
                                            pooling_s=(2, 1), pool_kwargs=dict(padding=(0, 1)), norm=None)
        block_6 = BlocksTorch.conv_norm_act(nm[5], nm[6], k=ks[6], s=ss[6], p=ps[6], act=activation, index=6,
                                            pooling=None, norm='bn')
        self.cnn = nn.Sequential(block_0, block_1, block_2, block_3, block_4, block_5, block_6)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # 防止出现梯度为nan/inf的情况，将这种梯度置为0
