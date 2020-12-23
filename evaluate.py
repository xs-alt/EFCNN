import os
import glob
import time
from tqdm import tqdm
import argparse
from random import shuffle

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian

from dataset_processing.grasp import BoundingBoxes, detect_grasps
from datasets import Cornell
from model import Model


def plot_output(rgb_img, depth_img, grasp_position_img, grasp_angle_img, ground_truth_bbs, no_grasps=1, grasp_width_img=None):
    """
    Visualize the outputs.
    """
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    grasp_position_img = gaussian(grasp_position_img, 5.0, preserve_range=True)

    if grasp_width_img is not None:
        grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)

    gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs)
    gs = detect_grasps(grasp_position_img, grasp_angle_img,
                       width_img=grasp_width_img, no_grasps=no_grasps, ang_threshold=0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax, color='r')

    # for g in gt_bbs:
    #    g.plot(ax, color='g')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap=plt.cm.binary)
    for g in gs:
        g.plot(ax, color='r')

    # for g in gt_bbs:
    #    g.plot(ax, color='g')

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(grasp_position_img, cmap='Reds', vmin=0, vmax=1)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv',
                     vmin=-np.pi / 2, vmax=np.pi / 2)
    plt.colorbar(plot)
    plt.show()


def calculate_iou_matches(grasp_positions_out, grasp_angles_out, ground_truth_bbs, no_grasps=1, grasp_width_out=None, min_iou=0.40):
    """
    Calculate a success score using the (by default) 25% IOU metric.
    Note that these results don't really reflect real-world performance.
    """
    succeeded = []
    failed = []
    for i in range(grasp_positions_out.shape[0]):  # batch size
        grasp_position = grasp_positions_out[i, ].squeeze()
        grasp_angle = grasp_angles_out[i, :, :].squeeze()

        grasp_position = gaussian(grasp_position, 5.0, preserve_range=True)

        if grasp_width_out is not None:
            grasp_width = grasp_width_out[i, ].squeeze()
            grasp_width = gaussian(grasp_width, 1.0, preserve_range=True)
        else:
            grasp_width = None

        gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs[i, ].squeeze())
        gs = detect_grasps(grasp_position, grasp_angle,
                           width_img=grasp_width, no_grasps=no_grasps, ang_threshold=0)
        for g in gs:
            if g.max_iou(gt_bbs) > min_iou:
                succeeded.append(i)
                break
            else:
                failed.append(i)

    return succeeded, failed


def run(network):
    global NO_GRASPS, VISUALIZE_FAILURES, VISUALIZE_SUCCESSES
    with torch.no_grad():
        succeed_count = 0
        fail_count = 0
        net = torch.load(network)
        model = Model()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(net)
        model = model.eval()
        model = model.cuda()

        for i, _ in enumerate(tqdm(cornell)):
            rgb, depth, grasp_points_img, cos_img, sin_img, grasp_width, bbs = cornell[i]
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
                pred_sin.detach().cpu().numpy(), pred_cos.detach().cpu().numpy())/2.0
            grasp_angles_out = np.transpose(grasp_angles_out, (0, 2, 3, 1))
            grasp_width_out = pred_width.detach().cpu().numpy() * 150.0
            grasp_width_out = np.transpose(grasp_width_out, (0, 2, 3, 1))

            succeeded, failed = calculate_iou_matches(
                grasp_positions_out, grasp_angles_out, bbs_all, no_grasps=NO_GRASPS, grasp_width_out=grasp_width_out)
            succeed_count += len(succeeded)
            fail_count += len(failed)

            if VISUALIZE_FAILURES:
                shuffle(failed)
                for i in failed:
                    print('Plotting Failures')
                    plot_output(rgb_imgs, depth_imgs[i, ], grasp_positions_out[i, ].squeeze(), grasp_angles_out[i, ].squeeze(), bbs_all[i, ],
                                no_grasps=NO_GRASPS, grasp_width_img=grasp_width_out[i, ].squeeze())

            if VISUALIZE_SUCCESSES:
                shuffle(succeeded)
                for i in succeeded:
                    print('Plotting Successes')
                    plot_output(rgb_imgs, depth_imgs[i, ], grasp_positions_out[i, ].squeeze(), grasp_angles_out[i, ].squeeze(), bbs_all[i, ],
                                no_grasps=NO_GRASPS, grasp_width_img=grasp_width_out[i, ].squeeze())

        print('succeed', succeed_count*1.0/(succeed_count+fail_count))

    return succeed_count*1.0/(succeed_count+fail_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='/media/ros/0D1416760D141676/Documents/CORNELL/datasets',
                        help='datasets path')
    parser.add_argument('--model', type=str, default='',
                        help='path to model file')
    parser.add_argument('--num_grasps', type=int, default=1,
                        help='number of grasps to evaluate')
    parser.add_argument('--vis_failures', type=bool, default=True,
                        help='visualize failure grasps')
    parser.add_argument('--vis_successes', type=bool, default=False,
                        help='visualize success grasps')
    opt = parser.parse_args()
    NO_GRASPS = opt.num_grasps
    VISUALIZE_FAILURES = opt.vis_failures
    VISUALIZE_SUCCESSES = opt.vis_successes
    cornell = Cornell(opt.datasets, train=False)
    run(opt.model)
