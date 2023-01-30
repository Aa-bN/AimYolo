# -*- coding = utf-8 -*-
# @Time : 2022/10/17 15:21
# @Author : cxk
# @File : z_captureScreen.py
# @Software : PyCharm

import os
import time
import cv2

import numpy as np

from matplotlib import pyplot as plt
from mss import mss


def capScreen(mss_obj, bound_box):
    '''
    mss_obj: mss object
    bound_box: dict, 截图区域，eg.{'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
    return: numpy_array
    (maybe add saves arg.)
    '''
    sct_img = mss_obj.grab(bound_box)
    sct_img = np.array(sct_img)         # HWC, and C==4
    sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR)     # 4 channels -> 3 channels
    return sct_img


def capSave(img_array, img_name, save_path):
    cv2.imwrite(os.path.join(save_path, img_name), img_array)


def testTime():
    mss_path = r'./saves/mss_saves'  # 截屏存储目录
    if not os.path.exists(mss_path):
        os.mkdir(mss_path)

    bounding_box = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}  # 截图区域，距离屏幕左上角距离，以及截屏宽高

    sct = mss()  # mss对象
    counter = 0
    img_name = '0.jpg'
    start = time.time()
    while True:
        sct_array = capScreen(sct, bounding_box)
        counter = counter + 1
        # img_name = str(counter) + '.jpg'
        # cv2.imwrite(os.path.join(mss_path, img_name), sct_array)
        if counter == 99:
            break

    end = time.time()
    print('time_cost:', end - start, 's')


def testPicFormat():
    sct = mss()
    bounding_box = {'left': 0, 'top': 0, 'width': 512, 'height': 256}
    img_array = capScreen(sct, bounding_box)        # img_array -> [256, 512, 4], this is HWC, need to be CHW
    plt.imshow(img_array)                           # BGR, need to be RGB
    plt.show()


if __name__ == '__main__':

    testTime()
    testPicFormat()




