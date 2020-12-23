#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import sys
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
from image_wise_datasets import Cornell
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
parser.add_argument(
    '--outf',
    type=str,
    default='five_fold_0',
    help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--kf', type=int, default='0', help='k-fold number')


opt = parser.parse_args()
print(opt)

opt.manualSeed = 1000  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = Cornell(opt.datasets, train=True, kf_num=opt.kf)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(
        opt.workers))
test_dataset = Cornell(opt.datasets, train=False, kf_num=opt.kf)
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


def blue(x): return '\033[94m' + x + '\033[0m'


# load model
classifier = Model()
classifier = torch.nn.DataParallel(classifier)
classifier.cuda()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters())
num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(tqdm(dataloader), 0):
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

        if i % 100 == 0:
            with torch.no_grad():
                j, data = next(enumerate((test_dataloader), 0))
                rgb, depth, pos, cos, sin, width, bbs = data
                rgb, depth, pos, cos, sin, width = Variable(rgb), Variable(
                    depth), Variable(pos), Variable(cos), Variable(sin), Variable(width)
                rgb, depth, pos, cos, sin, width = rgb.cuda(), depth.cuda(
                ), pos.cuda(), cos.cuda(), sin.cuda(), width.cuda()
                classifier = classifier.eval()
                pred_pos, pred_cos, pred_sin, pred_width = classifier(depth)
                pos_loss = F.mse_loss(pred_pos.float(), pos.float()) * 5
                cos_loss = F.mse_loss(pred_cos.float(), cos.float()) * 3
                sin_loss = F.mse_loss(pred_sin.float(), sin.float()) * 3
                width_loss = F.mse_loss(pred_width.float(), width.float()) * 4
                loss = pos_loss + cos_loss + sin_loss + width_loss
                print(
                    '[%d: %d/%d] %s val loss: %f, pos loss: %f, cos_loss: %f, sin_loss: %f, width_loss: %f ' %
                    (epoch,
                     i,
                     num_batch,
                     blue('test'),
                        loss.item(),
                        pos_loss.item(),
                        cos_loss.item(),
                        sin_loss.item(),
                        width_loss.item()))
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

    torch.save(classifier.state_dict(), '%s/model_%d.pth' % (opt.outf, epoch))
