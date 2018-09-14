#coding: 'utf-8'

"""
LDP_Net
Local_Depth_Dataset

created by Kazunari on 2018/08/23 
"""

import numpy as np
import os
import logging
import pickle
import sys

import chainer
import cv2

class LocalDepthDataset(chainer.dataset.DatasetMixin):

    # Dataset Unique Values for Transform
    mean_color = np.array([110.43808, 116.54863, 125.91209], np.float32)
    image_mean = 117.63293
    image_stddev = 66.46351
    sun_image_size = (480, 640)
    sun_down_sampling_size = (240, 320)
    sun_input_size = (228, 304)
    sun_output_size = (109, 147)
    sun_predicted_region = (11, 13, 216, 292)
    sun_depth_scale = 10000

    eigen_depth_scale = 0.1

    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.mode = mode

        with open(os.path.join(data_dir, "roi_list.pkl"), "rb") as f:
            roi_data = pickle.load(f)
            if self.mode in ("train", "test"):
                roi_list = roi_data[self.mode]
            else:
                logging.error("Set mode to \"train\" or \"test\".")
                sys.exit(-1)

        with open(os.path.join(data_dir, "class_ids.pkl"), "rb") as f:
            class_ids = pickle.load(f)

        self.rois = roi_list
        self.class_ids = class_ids

    def __len__(self):
        return len(self.rois)

    def get_images(self, i):
        img_path = os.path.join(self.data_dir, self.rois[i]["image_path"])
        img = cv2.imread(img_path)
        img = np.asarray(img, dtype=np.float32)
        img = np.clip(img, 0, 255)

        pred_depth_path = os.path.join(self.data_dir, self.rois[i]["pred_depth_path"])
        pred_depth = np.load(pred_depth_path) * self.eigen_depth_scale

        depth_path = os.path.join(self.data_dir, self.rois[i]["depth_path"])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = np.asarray(depth, dtype=np.float32)
        depth = depth / self.sun_depth_scale  # PNG value -> depth value (unit : meter)

        return img, pred_depth, depth

    def get_example(self, i):
        if i >= len(self):
            raise IndexError("index is too large")

        roi = self.rois[i]["2DBB"]
        class_id = self.rois[i]["class_id"] - 1  # 1 ~ n -> 0 ~ n-1

        img, pred_depth, depth = self.get_images(i)

        return roi, class_id, img, pred_depth, depth

    def get_class_id_size(self):
        return len(self.class_ids)  # classes