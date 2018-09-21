#coding: 'utf-8'

"""
LDP_Net
visualize_depth

created by Kazunari on 2018/09/17 
"""

from __future__ import division

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
sys.path.append(".")
import numpy as np
import os.path as osp
import os
import cv2
import pickle
import datetime

import matplotlib

import chainer
from chainerui.extensions import CommandsExtension

chainer.set_debug(True)

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from model.ldp_net import LDP_Net
from model.ldp_net_train_chain import LDPNetTrainChain
from dataset.Local_Depth_Dataset import LocalDepthDataset
from dataset.LDD_Transform import LDDTransform
from evaluation.ldp_evaluator import LDPNetEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Training script for LDP Net"
    )
    parser.add_argument('--dataset_path', '-p', type=str,
                        default="/home/takagi.kazunari/projects/datasets/LocalDepthDataset_v2")
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--pretrained_model', '-m', default=None)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--out', '-o', default='test_result',
                        help='Root directory of output')
    parser.add_argument('--dir', '-d', default=None,
                        help='Output directory')
    parser.add_argument('--normalize_depth', '-n', action='store_true')
    args = parser.parse_args()

    if args.out is not "test_result":
        out_dir = args.out
    elif args.out is "test_result" and args.dir is not None:
        out_dir = osp.join(args.out, args.dir)
    else:
        import datetime
        out_dir = osp.join(args.out, "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now()))

    out_dir = os.path.join(out_dir, "images")

    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    laina_dir = os.path.join(out_dir, "laina")
    gt_dir = os.path.join(out_dir, "gt")

    os.makedirs(out_dir, exist_ok=True)

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(laina_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    test_data = LocalDepthDataset(args.dataset_path, mode="test")
    transform = LDDTransform(test_data)

    ldp_net = LDP_Net(f_size=64,
                      rgbd_channel=4,
                      pretrained_model=args.pretrained_model)

    model = LDPNetTrainChain(ldp_net)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    for i in range(len(test_data)):
        roi, class_id, img, pred_depth, depth = test_data.get_example(i)

        roi_img, roi_depth, roi_pred_depth, t_roi = transform.get_cropped_roi_data(img, depth, pred_depth, roi)
        in_img, in_depth, in_pred_depth = transform.resize_to_input(roi_img, roi_depth, roi_pred_depth)
        in_img = ((in_img - transform.image_mean) / transform.image_stddev)

        # Create Mask
        eps = np.finfo(np.float32).eps
        mask = eps <= in_depth

        in_img = chainer.cuda.to_gpu(np.expand_dims(in_img, axis=0))
        in_depth = chainer.cuda.to_gpu(in_depth)
        in_pred_depth = chainer.cuda.to_gpu(np.expand_dims(in_pred_depth, axis=0))
        mask = chainer.cuda.to_gpu(mask)

        y = model.ldp_net(in_img, in_pred_depth, None)

        y = chainer.cuda.to_cpu(y.data)
        y = y[0, 0].transpose(1, 0)
        y = cv2.resize(y, (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        y = ((y - y.min()) / (y.max() - y.min())) * 255
        y = np.asarray(y, dtype=np.uint8)
        cv2.imwrite(osp.join(depth_dir, str(i+1).zfill(6) + ".png"), y)

        cv2.imwrite(osp.join(rgb_dir, str(i+1).zfill(6) + ".png"), roi_img)

        roi_depth = ((roi_depth - roi_depth.min()) / (roi_depth.max() - roi_depth.min())) * 255
        cv2.imwrite(osp.join(gt_dir, str(i+1).zfill(6) + ".png"), roi_depth)

        in_pred_depth = chainer.cuda.to_cpu(in_pred_depth)
        in_pred_depth = in_pred_depth[0, 0].transpose(1, 0)
        in_pred_depth = cv2.resize(in_pred_depth, (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        in_pred_depth = ((in_pred_depth - in_pred_depth.min()) / (in_pred_depth.max() - in_pred_depth.min())) * 255
        cv2.imwrite(osp.join(laina_dir, str(i+1).zfill(6) + ".png"), in_pred_depth)


if __name__ == "__main__":
    main()