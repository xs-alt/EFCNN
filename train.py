#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from datasets import Cornell
from evaluate import calculate_iou_matches
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument(
    '--datasets',
    type=str,
    default='/media/ros/0D1416760D141676/Documents/CORNELL/datasets',
    help='datasets path')
parser.add_argument(
    '--batchSize',
    type=int,
    default=16,
    help='input batch size')
parser.add_argument(
    '--workers',
    type=int,
    help='number of data loading workers',
    default=0)
parser.add_argument('--nepoch', type=int, default=500,
                    help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='out', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--num_grasps', type=int, default=5,
                    help='number of grasps to evaluate')


opt = parser.parse_args()
print(opt)

# Number of local maxima to check against ground truth grasps.
NO_GRASPS = opt.num_grasps

opt.manualSeed = 1000  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = Cornell(opt.datasets, train=True)
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(
        opt.workers))

test_dataset = Cornell(opt.datasets, train=False)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(
        opt.workers))

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass


# load model
classifier = Model()
classifier = torch.nn.DataParallel(classifier)
classifier.cuda()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters())
num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(tqdm(train_dataloader), 0):
        rgb, depth, pos, cos, sin, width = data
        raw_angle = np.arctan2(sin.numpy(), cos.numpy()) / 2.0
        rgb, depth, pos, cos, sin, width = Variable(rgb), Variable(
            depth), Variable(pos), Variable(cos), Variable(sin), Variable(width)
        rgb, depth, pos, cos, sin, width = rgb.cuda(), depth.cuda(
        ), pos.cuda(), cos.cuda(), sin.cuda(), width.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred_pos, pred_cos, pred_sin, pred_width = classifier(depth)
        grasp_angles_out = np.arctan2(
            pred_sin.detach().cpu().numpy(),
            pred_cos.detach().cpu().numpy()) / 2.0
        pos_loss = F.mse_loss(pred_pos.float(), pos.float()) * 5
        cos_loss = F.mse_loss(pred_cos.float(), cos.float()) * 3
        sin_loss = F.mse_loss(pred_sin.float(), sin.float()) * 3
        width_loss = F.mse_loss(pred_width.float(), width.float()) * 4
        loss = pos_loss + cos_loss + sin_loss + width_loss
        loss.backward()
        optimizer.step()
    print(
        '[%d: %d/%d] train loss: %f, pos loss: %f, cos_loss: %f, sin_loss: %f, width_loss: %f ' %
        (epoch,
         i,
         num_batch,
         loss.item(),
         pos_loss.item(),
         cos_loss.item(),
         sin_loss.item(),
         width_loss.item()))

    with torch.no_grad():
        succeed_count = 0
        fail_count = 0
        model = classifier.eval()

        for i in range(len(test_dataset)):
            rgb, depth, grasp_points_img, cos_img, sin_img, grasp_width, bbs = test_dataset[
                i]
            rgb_imgs = rgb
            depth_imgs = depth
            bbs = np.expand_dims(bbs, -1)
            bbs_all = bbs
            bbs_all = np.transpose(bbs_all, (3, 0, 1, 2))

            input_data = depth
            input_data = np.expand_dims(input_data, 0)
            input_data = torch.tensor(input_data)
            input_data = input_data.cuda()

            pred_pos, pred_cos, pred_sin, pred_width = model(input_data)

            grasp_positions_out = pred_pos.detach().cpu().numpy()
            grasp_positions_out = np.transpose(
                grasp_positions_out, (0, 2, 3, 1))
            grasp_angles_out = np.arctan2(
                pred_sin.detach().cpu().numpy(),
                pred_cos.detach().cpu().numpy()) / 2.0
            grasp_angles_out = np.transpose(grasp_angles_out, (0, 2, 3, 1))
            grasp_width_out = pred_width.detach().cpu().numpy() * 150.0
            grasp_width_out = np.transpose(grasp_width_out, (0, 2, 3, 1))

            succeeded, failed = calculate_iou_matches(
                grasp_positions_out, grasp_angles_out, bbs_all, no_grasps=NO_GRASPS, grasp_width_out=grasp_width_out)
            succeed_count += len(succeeded)
            fail_count += len(failed)

        rate = succeed_count * 1.0 / (succeed_count + fail_count)
        print('model: {}, success rate: {}'.format(epoch, rate))

    torch.save(
        classifier.state_dict(), '%s/model_%d-%f.pth' %
        (opt.outf, epoch, rate))
