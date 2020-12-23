#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

import os
import numpy as np
import pickle
import sys


TRAIN_SPLIT = 0.8


class Cornell(data.Dataset):
    def __init__(self, root, train=True):
        self.data_path = root
        self.train = train
        self.files = os.listdir(self.data_path)

        if train:
            train_files = self.files[:int(len(self.files) * TRAIN_SPLIT)]
            self.num = len(train_files)
            self.files = train_files
        else:
            test_files = self.files[int(len(self.files) * TRAIN_SPLIT):]
            self.num = len(test_files)
            self.files = test_files

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        meta = self.files[index]

        with open(os.path.join(self.data_path, meta), 'rb') as handle:
            if sys.version_info.major > 2:
                file = pickle.load(handle, encoding='latin1')
            else:
                file = pickle.load(handle)

            rgb = np.asarray(file['rgb'])
            rgb = np.squeeze(rgb, axis=0)

            depth = np.asarray(file['depth_inpainted'])
            depth = np.squeeze(depth, axis=0)

            pos_img = np.asarray(file['grasp_points_img'])
            pos_img = np.squeeze(pos_img, axis=0)

            angle_img = np.asarray(file['angle_img'])
            angle_img = np.squeeze(angle_img, axis=0)

            width_img = np.asarray(file['grasp_width'])
            width_img = np.squeeze(width_img, axis=0)

            bbs = np.asarray(file['bounding_boxes'])
            bbs = np.squeeze(bbs, axis=0)

            rgb = torch.tensor(rgb)
            depth = torch.tensor(depth)
            grasp_points_img = torch.tensor(pos_img)
            grasp_width = torch.tensor(width_img)
            cos_img = np.cos(2 * angle_img)
            sin_img = np.sin(2 * angle_img)
            cos_img = torch.tensor(cos_img)
            sin_img = torch.tensor(sin_img)

            if self.train is False:
                bbs = torch.tensor(bbs)
                return rgb, depth, grasp_points_img, cos_img, sin_img, grasp_width, bbs
            else:
                return rgb, depth, grasp_points_img, cos_img, sin_img, grasp_width


if __name__ == '__main__':
    cornell = Cornell(
        '/media/ros/0D1416760D141676/Documents/CORNELL/datasets', train=False)
    print(len(cornell))
