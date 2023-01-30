# # -*- coding = utf-8 -*-
# # @Time : 2022/10/17 20:45
# # @Author : cxk
# # @File : utils_new.py
# # @Software : PyCharm
#
# import math
# import os
# import re
# from copy import copy
# from pathlib import Path
#
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sn
# import torch
# from PIL import Image, ImageDraw, ImageFont
#
# RANK = int(os.getenv('RANK', -1))
#
#
# def is_ascii(s=''):
#     # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
#     s = str(s)  # convert list, tuple, None, etc. to str
#     return len(s.encode().decode('ascii', 'ignore')) == len(s)
#
#
# def is_chinese(s='人工智能'):
#     # Is string composed of any Chinese characters?
#     return True if re.search('[\u4e00-\u9fff]', str(s)) else False
#
#
# def check_pil_font(font=FONT, size=10):
#     # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
#     font = Path(font)
#     font = font if font.exists() else (CONFIG_DIR / font.name)
#     try:
#         return ImageFont.truetype(str(font) if font.exists() else font.name, size)
#     except Exception:  # download if missing
#         check_font(font)
#         try:
#             return ImageFont.truetype(str(font), size)
#         except TypeError:
#             check_requirements('Pillow>=8.4.0')  # known issue https://github.com/ultralytics/yolov5/issues/5374
#
#
# class Colors:
#     # Ultralytics color palette https://ultralytics.com/
#     def __init__(self):
#         # hex = matplotlib.colors.TABLEAU_COLORS.values()
#         hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
#                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
#         self.palette = [self.hex2rgb('#' + c) for c in hex]
#         self.n = len(self.palette)
#
#     def __call__(self, i, bgr=False):
#         c = self.palette[int(i) % self.n]
#         return (c[2], c[1], c[0]) if bgr else c
#
#     @staticmethod
#     def hex2rgb(h):  # rgb order (PIL)
#         return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
#
#
# colors = Colors()  # create instance for 'from utils.plots import colors'
#
# class Annotator:
#     if RANK in (-1, 0):
#         check_pil_font()  # download TTF if necessary
#
#     # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
#     def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
#         assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
#         self.pil = pil or not is_ascii(example) or is_chinese(example)
#         if self.pil:  # use PIL
#             self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
#             self.draw = ImageDraw.Draw(self.im)
#             self.font = check_pil_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
#                                        size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
#         else:  # use cv2
#             self.im = im
#         self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
#
#     def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
#         # Add one xyxy box to image with label
#         if self.pil or not is_ascii(label):
#             self.draw.rectangle(box, width=self.lw, outline=color)  # box
#             if label:
#                 w, h = self.font.getsize(label)  # text width, height
#                 outside = box[1] - h >= 0  # label fits outside box
#                 self.draw.rectangle((box[0],
#                                      box[1] - h if outside else box[1],
#                                      box[0] + w + 1,
#                                      box[1] + 1 if outside else box[1] + h + 1), fill=color)
#                 # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
#                 self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
#         else:  # cv2
#             p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#             cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
#             if label:
#                 tf = max(self.lw - 1, 1)  # font thickness
#                 w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
#                 outside = p1[1] - h - 3 >= 0  # label fits outside box
#                 p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#                 cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
#                 cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
#                             thickness=tf, lineType=cv2.LINE_AA)
#
#     def rectangle(self, xy, fill=None, outline=None, width=1):
#         # Add rectangle to image (PIL-only)
#         self.draw.rectangle(xy, fill, outline, width)
#
#     def text(self, xy, text, txt_color=(255, 255, 255)):
#         # Add text to image (PIL-only)
#         w, h = self.font.getsize(text)  # text width, height
#         self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)
#
#     def result(self):
#         # Return annotated image as array
#         return np.asarray(self.im)
#
#
