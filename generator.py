# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 11:52
# @Author  : gpwang
# @File    : generator.py
# @Software: PyCharm
"""
字符图片生成器
"""
import random

import cv2
import numpy as np
# 生成随机的颜色
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data.dataset import Dataset

from data.words import Word
from fontutils import FONT_CHARS_DICT

word = Word()


def random_color(lower_val, upper_val):
    return [random.randint(lower_val, upper_val),
            random.randint(lower_val, upper_val),
            random.randint(lower_val, upper_val)]


def put_text(image, x, y, text, font, color=None):
    """
    写中文字
    :param image:
    :param x:
    :param y:
    :param text: 文字
    :param font:
    :param color:
    :return:
    """
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    color = (255, 0, 0) if color is None else color
    draw.text((x, y), text, color, font=font)
    return np.array(im)


class Generator(Dataset):
    def __init__(self, alpha, direction='horizontal'):
        """

        :param alpha: 所有字符
        :param direction:
        """
        super(Generator, self).__init__()
        self.alpha = alpha
        self.direction = direction
        self.alpha_list = list(alpha)
        self.min_len = 5
        self.max_len_list = [16, 19, 24, 26]
        self.max_len = max(self.max_len_list)
        self.font_size_list = [30, 25, 20, 18]
        self.font_path_list = list(FONT_CHARS_DICT.keys())
        self.font_list = []
        for size in self.font_size_list:
            self.font_list.append(
                [ImageFont.truetype(font_path, size=size) for font_path in self.font_path_list])
        if self.direction == "horizontal":
            self.im_h = 32
            self.im_w = 200
        else:
            self.im_h = 512
            self.im_w = 32

    def gen_background(self):
        """
        随机生成背景
        :return:
        """
        a = random.random()
        pure_bg = np.ones((self.im_h, self.im_w, 3)) * np.array(random_color(0, 100))
        random_bg = np.random.rand(self.im_h, self.im_w, 3) * 100
        if a < 0.1:
            return random_bg
        elif a < 0.8:
            return pure_bg
        else:
            b = random.random()
            mix_bg = b * pure_bg + (1 - b) * random_bg
            return mix_bg

    def horizontal_draw(self, draw, text, font, color, char_w, char_h):
        """

        :param draw:
        :param text: 文本
        :param font: 字体
        :param color: 颜色
        :param char_w: 字符宽度
        :param char_h: 字符高度
        :return:
        """
        text_w = len(text) * char_w  # 计算字符的总宽度
        h_margin = max(self.im_h - char_h, 1)
        w_margin = max(self.im_w - char_w, 1)
        x_shift = np.random.randint(0, w_margin)
        y_shift = np.random.randint(0, h_margin)
        i = 0
        while i < len(text):
            draw.text((x_shift, y_shift), text[i], color, font=font)
            i += 1
            x_shift += char_w
            y_shift = np.random.randint(0, h_margin)
            # 如果下个字符超出图像，则退出
            if x_shift + char_w > self.im_w:
                break
        return text[:i]

    def vertical_draw(self, draw, text, font, color, char_w, char_h):
        """
        垂直方向文字生成
        :param draw:
        :param text:
        :param font:
        :param color:
        :param char_w:
        :param char_h:
        :return:
        """
        text_h = len(text) * char_h
        h_margin = max(self.im_h - text_h, 1)
        w_margin = max(self.im_w - char_w, 1)
        x_shift = np.random.randint(0, w_margin)
        y_shift = np.random.randint(0, h_margin)
        i = 0
        while i < len(text):
            draw.text((x_shift, y_shift), text[i], color, font=font)
            i += 1
            x_shift = np.random.randint(0, w_margin)
            y_shift += char_h
            # 如果下个字符超出图像，则退出
            if y_shift + char_h > self.im_h:
                break
        return text[:i]

    def draw_text(self, draw, text, font, color, char_w, char_h):
        if self.direction == 'horizontal':
            return self.horizontal_draw(draw, text, font, color, char_w, char_h)
        return self.vertical_draw(draw, text, font, color, char_w, char_h)

    # 生成图片
    def gen_image(self):
        idx = np.random.randint(len(self.max_len_list))
        image = self.gen_background()
        image = image.astype(np.uint8)
        target_len = int(np.random.uniform(self.min_len, self.max_len_list[idx], size=1))

        # 随机选择size,font
        size_idx = np.random.randint(len(self.font_size_list))
        font_idx = np.random.randint(len(self.font_path_list))
        font = self.font_list[size_idx][font_idx]
        font_path = self.font_path_list[font_idx]
        # 在选中font字体的可见字符中随机选择target_len个字符
        # text = random.choices(FONT_CHARS_DICT[font_path], k=target_len)
        # text = np.random.choice(FONT_CHARS_DICT[font_path], target_len)
        text = random.choices(self.alpha[1:], k=target_len)
        text = ''.join(text)
        # 计算字体的w和h
        w, char_h = font.getsize(text)
        char_w = int(w / len(text))
        # 写文字，生成图像
        im = Image.fromarray(image)
        draw = ImageDraw.Draw(im)
        color = tuple(random_color(105, 255))
        text = self.draw_text(draw, text, font, color, char_w, char_h)
        target_len = len(text)
        # 对应的类别
        indices = np.array([self.alpha.index(c) for c in text])
        # 转为灰度图
        image = np.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 亮度反转
        if random.random() > 0.5:
            image = 255 - image
        return image, indices, target_len, text

    def __getitem__(self, item):
        image, indices, target_len, text = self.gen_image()
        if self.direction == "horizontal":
            image = np.transpose(image[:, :, np.newaxis], axes=(2, 1, 0))
        else:
            image = np.transpose(image[:, :, np.newaxis], axes=(2, 0, 1))
        # 标准化
        image = image.astype(np.float32) / 255
        image -= 0.5
        image /= 0.5

        target = np.zeros(shape=(self.max_len,), dtype=np.long)
        target[:target_len] = indices
        if self.direction == 'horizontal':
            input_len = self.im_w // 4 - 3
        else:
            input_len = self.im_w // 16 - 1
        return image, target, input_len, target_len

    def __len__(self):
        return len(self.alpha) * 100


def lll_image_gen(direction='horizontal'):
    gen = Generator(word.get_all_words()[:], direction=direction)
    for i in range(100):
        im, indices, target_len, text = gen.gen_image()
        cv2.imwrite("./images/{}-{:03d}.jpg".format(text, i + 1), im)

        print(''.join([gen.alpha[j] for j in indices]))


if __name__ == '__main__':
    lll_image_gen()
