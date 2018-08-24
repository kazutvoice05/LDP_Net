#coding: 'utf-8'

"""
LDP_Net
train

created by Kazunari on 2018/08/23 
"""

from __future__ import division

import argparse
import sys
import numpy as np
import os.path as osp
import datetime

import matplotlib

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger

from .model.ldp_net import LDP_Net
from .model.ldp_net_train_chain import LDPNetTrainChain


from .dataset.Local_Depth_Dataset import LocalDepthDataset

def main():
    parser = argparse.ArgumentParser(
        description="Training script for LDP Net"
    )
    parser.add_argument('--dataset_path', '-path', type=str,
                        default="/home/takagi.kazunari/projects/datasets/SUNRGBD_2DBB_fixed")
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--out', '-o', default='sunrgbd_result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_data = LocalDepthDataset(args.dataset_path, mode="train")
    test_data = LocalDepthDataset(args.dataset_path, mode="test")

    ldp_net = LDP_Net(f_size=64, input_channel=34)

    model = LDPNetTrainChain(ldp_net)
    if args.gpu >= 0:
        chainer.cuda.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    # TODO : Confirm that add_hook is needed.
    optimizer.add_hook(chainer.optimizer.optimizer_hooks.WeightDecay(rate=0.0005))

    train_data = TransformDataset(train_data, Transform(ldp_net))

    train_iter = chainer.iterators.SerialIterator(train_data, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batch_size,
                                                 repeat=False, shuffle=False)
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(
        extensions.snapshot_object(model.ldp_net, 'snapshot_model.npz'),
        trigger=(args.iteration, "iteration"))

