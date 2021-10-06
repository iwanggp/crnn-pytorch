#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@File   :pt2onnx.py
@IDE    :PyCharm
@Author :gpwang
@Date   :2021/10/6
@Desc   :将pt权重转换为onnx
=================================================='''
import torch
import models.crnn as crnn
import config
import numpy as np
import cv2

alphabet = config.alphabet


def pt2onnx(pt):
    nc = len(alphabet) + 1
    model = crnn.CRNN(32, 1, nc, 256)  # 转换的NHidden必须和训练模型一致
    model.load_state_dict(torch.load(pt))
    dummy_input = torch.randn(1, 1, 32, 280)
    torch.onnx.export(model, dummy_input, 'crnn.onnx', verbose=True)


def decodeText(scores):
    text = ""
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += "-"
    print(text)
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)


def test_onnx(onnx):
    recongnizer = cv2.dnn.readNetFromONNX(onnx)
    img = cv2.imread('trans.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale = img.shape[0] * 1.0 / 32
    w = img.shape[1] / scale
    blob = cv2.dnn.blobFromImage(img, size=(280, 32), mean=127.5, scalefactor=1 / 127.5)
    recongnizer.setInput(blob)
    result = recongnizer.forward()
    texxx = decodeText(result)
    print(texxx)
