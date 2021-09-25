# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 11:16
# @Author  : gpwang
# @File    : words.py
# @Software: PyCharm
import codecs


class Word(object):
    def __init__(self, chinese_word=True, alphabet=True, digit=True, punctuation=True, currency=True):
        """

        :param chinese_word: 中文
        :param alphabet: 字母
        :param digit: 数字
        :param punctuation: 标点符号
        :param currency: 货币符号
        """

        self.chinese_word = chinese_word
        self.alphabet = alphabet
        self.digit = digit
        self.punctuation = punctuation
        self.currency = currency

    @classmethod
    def get_digits(cls):
        return '0123456789'

    @classmethod
    def get_alphabet(cls):
        return 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    @classmethod
    def get_punctuations(cls):
        return "。，、；：？！…-·ˉˇ¨‘'“”～‖∶＂＇｀｜〃〔〕〈〉《》「」『』．.〖〗【】（）［］｛｝"

    @classmethod
    def get_currency(cls):
        return '$¥'

    def get_all_words(self):
        f = codecs.open('keys.txt', mode='r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        lines = [l.strip() for l in lines]
        return ' ' + ''.join(lines)


if __name__ == '__main__':
    w = Word()
    print(len(w.get_all_words()) == len(set(w.get_all_words())))
    print(w.get_all_words())
    print(w.get_all_words().__contains__(' '))
