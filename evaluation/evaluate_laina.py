#coding: 'utf-8'

"""
LDP_Net
evaluate_laina

created by Kazunari on 2018/09/18 
"""

from __future__ import division

import matplotlib
matplotlib.use("Agg")

import argparse
import pickle
import logging
import sys
sys.path.append(".")
import glob
import numpy as np
import os.path as osp
import os
import datetime

import matplotlib
import copy

from metrics import compute_metrics
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


class LainaPredictions(chainer.dataset.DatasetMixin):

    # Dataset Unique Values for Transform
    mean_color = np.array([110.43808, 116.54863, 125.91209], np.float32)
    image_mean = 117.63293
    image_stddev = 66.46351
    image_size = (480, 640)
    down_sampling_size = (240, 320)
    sun_depth_scale = 10000

    def __init__(self, data_dir):
        self.data_dir = data_dir

        npy_paths = sorted(glob.glob(osp.join(data_dir, "npy", "*")))
        pred_paths = sorted(glob.glob(osp.join(data_dir, "predicted_depths", "*")))

        self.npy_paths = npy_paths
        self.d_paths = pred_paths

    def __len__(self):
        return len(self.npy_paths)

    def get_images(self, i):
        stacked = np.asarray(np.load(self.npy_paths[i]), dtype=np.float32)
        depth = stacked[3, :, :]
        depth = np.expand_dims(depth, axis=0).transpose(1, 2, 0)

        mask = stacked[4, :, :]
        mask = np.expand_dims(mask, axis=0).transpose(1, 2, 0)

        pred_depth = np.asarray(np.load(self.d_paths[i]), dtype=np.float32)
        pred_depth = np.expand_dims(pred_depth, axis=0).transpose(1, 2, 0)

        return pred_depth, depth, mask

    def get_example(self, i):
        if i >= len(self):
            raise IndexError("index is too large")

        pred_depth, depth, mask = self.get_images(i)

        return pred_depth, depth, mask


class LainaEvaluator(extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'val'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, device=None):
        super(LainaEvaluator, self).__init__(
            iterator, target, device=device)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            iter = iterator
        else:
            iter = copy.copy(iterator)

        results = {}
        for batch in iter:
            pred_depth, depth, mask = self.converter(batch, self.device)

            batch_result = compute_metrics(pred_depth, depth, mask)

            for key in batch_result.keys():
                if key in results.keys():
                    results[key].append(batch_result[key])
                else:
                    results[key] = [batch_result[key]]

        for key in results.keys():
            results[key] = np.mean(results[key])

        return results

if __name__ == "__main__":
    dataset = LainaPredictions("/Users/Kazunari/projects/datasets/LocalDepthDataset_v2/test")

    test_iter = chainer.iterators.SerialIterator(dataset,16, repeat=False, shuffle=False)

    ldp_net = LDP_Net()
    model = LDPNetTrainChain(ldp_net)

    evaluator = LainaEvaluator(test_iter, model.ldp_net)

    results = evaluator()

    print(results)

    with open(osp.join(dataset.data_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(results, f)
