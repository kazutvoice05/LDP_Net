#coding: 'utf-8'

"""
LDP_Net
view_dataset

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


cv2.namedWindow("rgb")
cv2.namedWindow("depth")
cv2.namedWindow("laina")

train_data = LocalDepthDataset("/Users/Kazunari/projects/datasets/LocalDepthDataset_v2")

transform = LDDTransform(train_data)

for i in range(len(train_data)):
    roi, class_id, img, pred_depth, depth = train_data.get_example(i)

    roi_img, roi_depth, roi_pred_depth, c_roi = transform.get_cropped_roi_data(img, depth, pred_depth, roi)

    roi_img, roi_depth, roi_pred_depth = transform.resize_to_input(roi_img, roi_depth, roi_pred_depth)

    roi_img = np.asarray(roi_img, dtype=np.uint8)

    roi_depth = ((roi_depth - roi_depth.min()) / (roi_depth.max() - roi_depth.min())) * 255
    roi_depth = np.asarray(roi_depth, dtype=np.uint8)

    roi_pred_depth = ((roi_pred_depth - roi_pred_depth.min()) / (roi_pred_depth.max() - roi_pred_depth.min())) * 255
    roi_pred_depth = np.asarray(roi_pred_depth, dtype=np.uint8)

    cv2.imshow("rgb", roi_img)
    cv2.imshow("depth", roi_depth)
    cv2.imshow("laina", roi_pred_depth)

    cv2.waitKey(1)

    print("d")