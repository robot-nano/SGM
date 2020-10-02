import torch.utils.data as data
import skimage
import skimage.io
import skimage.transform

from PIL import Image
import numpy as np
import random
from struct import unpack
import re
import sys
import matplotlib.pyplot as plt


def train_transform(temp_data, crop_height, crop_width, left_right=False, shift=0):
    _, h, w = np.shape(temp_data)

    if h > crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, h + shift, crop_width + shift], 'float32')
        temp_data[6:7, :, :] = 1000
        temp_data[:, h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height + shift, crop_width + shift], 'float32')
        temp_data[6:7, :, :] = 1000
        temp_data[:, crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = temp
        _, h, w = np.shape(temp_data)
    if shift > 0:
        start_x = random.randint(0, w - crop_width)
        shift_x = random.randint(-shift, shift)
        if shift_x + shift_x < 0 or shift_x + start_x + crop_width > w:
            shift_x = 0

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = random.randint(0, w - crop_width)
        start_y = random.randint(0, h - crop_height)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    if left_right:
        right = temp_data[0: 3, :, :]
        left = temp_data[3: 6, :, :]
        target = temp_data[7, :, :]
        return left, right, target
    else:
        left = temp_data[0: 3, :, :]
        right = temp_data[3: 6, :, :]
        target = temp_data[6, :, :]
        return left, right, target


def test_transform(temp_data, crop_height, crop_width, left_right=False):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([8, crop_height, crop_width], 'float32')
        temp_data[6, :, :] = 1000
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = (w - crop_width) / 2
        start_y = (h - crop_height) / 2
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left = temp_data[0: 3, :, :]
    right = temp_data[3: 6, :, :]
    target = temp_data[6, :, :]

    return left, right, target


def load_kitti_data(file_path, current_file):
    """ load current file from the list """
    filename = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ/' + current_file[0: len(current_file) - 1]

    disp_left = Image.open(filename)
    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], "float32")
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]

    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]

    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    temp_data[6:7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.

    return temp_data


def load_kitti2015_data(file_path, current_file):
    """ load current file from the list """
    filename = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]

    disp_left = Image.open(filename)
    size = np.shape(left)

    height = size[0]
    width = size[1]
    temp_data = np.zeros([8, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    disp_left = np.asarray(disp_left)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]

    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]

    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    temp_data[6:7, :, :] = width * 2
    temp_data[6, :, :] = disp_left[:, :]
    temp = temp_data[6, :, :]
    temp[temp < 0.1] = width * 2 * 256
    temp_data[6, :, :] = temp / 256.

    return temp_data


class DatasetFromList(data.Dataset):
    def __init__(self, data_path, file_list, crop_size=[256, 256], training=True, left_right=False, kitti=False,
                 kitti2015=False, shift=0):
        super(DatasetFromList, self).__init__()
        f = open(file_list, 'r')
        self.data_path = data_path
        self.file_list = f.readlines()
        self.training = training
        self.crop_height = crop_size[0]
        self.crop_width = crop_size[1]
        self.left_right = left_right
        self.kitti = kitti
        self.kitti2015 = kitti2015
        self.shift = shift

    def __getitem__(self, index):
        if self.kitti:  # load kitti dataset
            temp_data = load_kitti_data(self.data_path, self.file_list[index])
        elif self.kitti2015:  # load kitti2015 dataset
            temp_data = load_kitti2015_data(self.data_path, self.file_list[index])
        else:
            print("only support kitti dataset")
            exit(-1)
        if self.training:
            input1, input2, target = train_transform(temp_data, self.crop_height, self.crop_width, self.left_right,
                                                     self.shift)
            return input1, input2, target
        else:
            pass

    def __len__(self):
        return len(self.file_list)
