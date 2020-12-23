import glob
import os
import sys 
import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pickle

sys.path.insert(0, '.')
from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp





parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_dir',
    type=str,
    default='/media/ros/0D1416760D141676/Documents/CORNELL/multi_datasets',
    help='output datasets path')
parser.add_argument(
    '--datasets',
    type=str,
    default='/home/ros/Documents/Grasp/grasp_multiObject/rgbd',
    help='raw datasets path')
parser.add_argument(
    '--image_size',
    type=int,
    default=400,
    help='dataset image size')
parser.add_argument(
    '--visualize_only',
    type=bool,
    default=True,
    help='if only do visualization')
opt = parser.parse_args()
OUTPUT_DIR = opt.output_dir
RAW_DATA_DIR = opt.datasets
OUTPUT_IMG_SIZE = (opt.image_size, opt.image_size)
VISUALIZE_ONLY = opt.visualize_only


# File name patterns for the different file types.  _ % '<image_id>'
_rgb_pattern = os.path.join(RAW_DATA_DIR, 'rgb_%s.jpg')
_depth_pattern = os.path.join(RAW_DATA_DIR, 'depth_%s.png')
_pos_grasp_pattern = os.path.join(RAW_DATA_DIR, 'rgb_%s_annotations.txt')


def get_image_ids():
    # Get all the input files, extract the numbers.
    rgb_images = glob.glob(_rgb_pattern % '*')
    rgb_images.sort()
    return [r[-8:-4] for r in rgb_images]


if __name__ == '__main__':
    # Create the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fields = [
        'img_id',
        'rgb',
        'depth_inpainted',
        'bounding_boxes',
        'grasp_points_img',
        'angle_img',
        'grasp_width'
    ]

    # Empty datatset.
    dataset = {
        'items': dict([(f, []) for f in fields])
    }

    ds = dataset['items']
    n = 0
    for img_id in get_image_ids():
        print('Processing: %s' % img_id)

        # Load the image
        rgb_img_base = Image(io.imread(_rgb_pattern % img_id))
        depth_img_base = DepthImage(io.imread(_depth_pattern % img_id))
        depth_img_base.inpaint()

        # Load Grasps.
        bounding_boxes_base = grasp.BoundingBoxes.load_from_file(
            _pos_grasp_pattern % img_id)
        center = bounding_boxes_base.center

        # new add
        rgb = rgb_img_base
        depth = depth_img_base
        bbs = bounding_boxes_base.copy()

        left = max(0, min(center[1] -
                          OUTPUT_IMG_SIZE[1] //
                          2, rgb.shape[1] -
                          OUTPUT_IMG_SIZE[1]))
        right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

        top = max(0, min(center[0] -
                         OUTPUT_IMG_SIZE[0] //
                         2, rgb.shape[0] -
                         OUTPUT_IMG_SIZE[0]))
        bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

        rgb.crop((top, left), (bottom, right))
        depth.crop((top, left), (bottom, right))
        bbs.offset((-top, -left))

        ds['img_id'] = []
        ds['rgb'] = []
        ds['depth_inpainted'] = []
        ds['bounding_boxes'] = []
        ds['grasp_points_img'] = []
        ds['angle_img'] = []
        ds['grasp_width'] = []
        n += 1

        depth.normalise()

        pos_img, ang_img, width_img = bbs.draw(depth.shape)

        if VISUALIZE_ONLY:
            f = plt.figure()
            ax = f.add_subplot(1, 5, 1)
            rgb.show(ax)
            bbs.show(ax)
            ax = f.add_subplot(1, 5, 2)
            depth.show(ax)
            bbs.show(ax)

            ax = f.add_subplot(1, 5, 3)
            ax.imshow(pos_img)

            ax = f.add_subplot(1, 5, 4)
            ax.imshow(ang_img)

            ax = f.add_subplot(1, 5, 5)
            ax.imshow(width_img)

            plt.show()
            continue

        else:
            rgb = rgb.img
            depth = np.expand_dims(depth.img, -1)
            grasp_points_img = np.expand_dims(pos_img, -1)
            grasp_angle_img = np.expand_dims(ang_img, -1)
            grasp_width = np.expand_dims(width_img, -1)
            grasp_width = np.clip(grasp_width, 0, 150) / 150.0
            bbs = bbs.to_array(pad_to=25)

            rgb = np.transpose(rgb, (2, 0, 1))
            depth = np.transpose(depth, (2, 0, 1))
            grasp_width = np.transpose(grasp_width, (2, 0, 1))
            grasp_angle = np.transpose(grasp_angle_img, (2, 0, 1))
            grasp_points_img = np.transpose(grasp_points_img, (2, 0, 1))

            ds['img_id'].append(int(img_id))
            ds['rgb'].append(rgb)
            ds['depth_inpainted'].append(depth)
            ds['bounding_boxes'].append(bbs)
            ds['grasp_points_img'].append(grasp_points_img)
            ds['grasp_width'].append(grasp_width)
            ds['angle_img'].append(grasp_angle)

            with open(os.path.join(OUTPUT_DIR, '{}.pickle'.format(n)), 'wb') as handle:
                pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
