# -*- coding = utf-8 -*-
# @Time : 2022/10/28 17:56
# @Author : cxk
# @File : z_detect5.py
# @Software : PyCharm


import sys
import ctypes
import signal

import argparse
import win32con
import win32api

from mss import mss
from pynput import mouse

from z_captureScreen import capScreen

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.utils import *

from z_ctypes import SendInput, mouse_input

PROCESS_PER_MONITOR_DPI_AWARE = 2
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)


def pre_process(img0, img_sz, half, device):
    """
    img0: from capScreen(), format: HWC, BGR
    """
    # padding resize
    img = letterbox(img0, new_shape=img_sz)[0]
    # convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)

    # preprocess
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0-255 to 0.0-1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def inference_img(img, model, augment, conf_thres, iou_thres, classes, agnostic):
    """
    推理，模型参数，...
    """
    # inference
    pred = model(img, augment=augment)[0]
    # apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic)
    return pred


def calculate_position(xyxy):
    """
    计算中心坐标
    """
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    # print('\n左上点坐标:(' + str(c1[0]) + ',' + str(c1[1]) + '), 右上点坐标:(' + str(c2[0]) + ',' + str(
    #     c1[1]) + ')')
    # print('左下点坐标:(' + str(c1[0]) + ',' + str(c2[1]) + '), 右下点坐标:(' + str(c2[0]) + ',' + str(
    #     c2[1]) + ')')
    # print("中心点的坐标为：(" + str((c2[0] - c1[0]) / 2 + c1[0]) + "," + str(
    #     (c2[1] - c1[1]) / 2 + c1[1]) + ")")
    center_x = int((c2[0] - c1[0]) / 2 + c1[0])
    center_y = int((c2[1] - c1[1]) / 2 + c1[1])
    return center_x, center_y


def view_imgs(img0):
    """
    弹窗展示结果，press q to quit
    """
    img0 = cv2.resize(img0, (480, 540))
    # img0 = cv2.resize(img0, (960, 540))
    cv2.imshow('ws demo', img0)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


def move_mouse(mouse_pynput, aim_persons_center):
    """
    move mouse，暂定为欧氏距离
    指针移动到距离当前鼠标位置最短的坐标处
    """
    if aim_persons_center:
        # 当前鼠标位置
        current_x, current_y = mouse_pynput.position
        # 距离当前位置最近的目标中心点and距离
        best_position = None
        for aim_person in aim_persons_center:
            # aim_person is a list
            dist = ((aim_person[0] - current_x) ** 2 + (aim_person[1] - current_y) ** 2) ** .5
            if not best_position:
                best_position = (aim_person, dist)
            else:
                _, old_dist = best_position
                if dist < old_dist:
                    best_position = (aim_person, dist)

        tx = int(best_position[0][0] / win32api.GetSystemMetrics(0) * 65535)
        ty = int(best_position[0][1] / win32api.GetSystemMetrics(1) * 65535)
        SendInput(mouse_input(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, tx, ty))


class AimYolo:

    def __init__(self, opt):
        self.weights = opt.weights
        self.img_size = opt.img_size
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.view_img = opt.view_img
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.augment = opt.augment

        # self.bounding_box = {'left': 0, 'top': 0, 'width': 960, 'height': 960}
        self.bounding_box = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}

        # load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.model = self.model.to(self.device)
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # name and color
        self.names = self.model.modules.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    @torch.no_grad()
    def run(self):

        img_sz = self.img_size

        # warm up
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

        # mss and capture screen
        sct = mss()
        print("The mss object created.")
        # mouse control
        mouse_control = mouse.Controller()
        print("The mouse controller created.")

        # 若只在循环中的if前定义空列表，在检测不到目标时，for循环没有对其进行初始化，会报错
        aim_persons_center = []
        aim_persons_center_head = []
        while True:
            img0 = capScreen(sct, self.bounding_box)  # HWC and BGR

            img = pre_process(img0=img0, img_sz=img_sz, half=self.half, device=self.device)

            t1 = torch_utils.time_synchronized()
            pred = inference_img(img=img, model=self.model, augment=self.augment, conf_thres=self.conf_thres,
                                 iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # process detections
            det = pred[0]

            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # write results
                aim_persons_center = []
                aim_persons_center_head = []
                for *xyxy, conf, cls in det:

                    # Add bbox to image
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)
                    center_x, center_y = calculate_position(xyxy)
                    aim_persons_center.append([center_x, center_y])
                    if int(cls) == 2 or int(cls) == 3:
                        aim_persons_center_head.append([center_x, center_y])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # 计算中心坐标，若有多个目标，只取其中一个
            # 取距离最近的一个
            move_mouse(mouse_control, aim_persons_center_head)
            aim_persons_center = []
            aim_persons_center_head = []

            # view img
            if self.view_img:
                view_imgs(img0=img0)

        # End ------------------------------------------


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best_200.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    opt = parseArgs()
    print(opt)

    aim_yolo = AimYolo(opt)
    print('The AimYolo Object Created.')

    aim_yolo.run()

