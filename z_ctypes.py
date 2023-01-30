# -*- coding = utf-8 -*-
# @Time : 2022/10/24 10:22
# @Author : cxk
# @File : z_ctypes.py
# @Software : PyCharm

import ctypes

LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort

INPUT_MOUSE = 0


class MouseInput(ctypes.Structure):
    _fields_ = [
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]


class InputUnion(ctypes.Union):
    _fields_ = [
        ('mi', MouseInput)
    ]


class Input(ctypes.Structure):
    _fields_ = [
        ('types', DWORD),
        ('iu', InputUnion)
    ]


def mouse_input_set(flags, x, y, data):
    return MouseInput(x, y, data, flags, 0, None)


def input_do(structure):
    if isinstance(structure, MouseInput):
        return Input(INPUT_MOUSE, InputUnion(mi=structure))
    raise TypeError('Cannot create Input structure!')


def mouse_input(flags, x=0, y=0, data=0):
    return input_do(mouse_input_set(flags, x, y, data))


def SendInput(*inputs):
    n_inputs = len(inputs)
    lp_input = Input * n_inputs
    p_inputs = lp_input(*inputs)
    cb_size = ctypes.c_int(ctypes.sizeof(Input))
    return ctypes.windll.user32.SendInput(n_inputs, p_inputs, cb_size)


if __name__ == '__main__':
    SendInput(mouse_input(1, -100, -200))


