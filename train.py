#coding: 'utf-8'

"""
LDP_Net
train

created by Kazunari on 2018/08/23 
"""

from __future__ import division

import matplotlib
matplotlib.use("Agg")

import argparse
import sys
sys.path.append(".")
import numpy as np
import os.path as osp
import datetime

import matplotlib

import chainer

chainer.set_debug(True)

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from model.ldp_net import LDP_Net
from model.ldp_net_train_chain import LDPNetTrainChain

from dataset.Local_Depth_Dataset import LocalDepthDataset
from dataset.LDD_Transform import LDDTransform

def main():
    parser = argparse.ArgumentParser(
        description="Training script for LDP Net"
    )
    parser.add_argument('--dataset_path', '-p', type=str,
                        default="/Users/Kazunari/projects/datasets/LocalDepthDataset")
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-5)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--out', '-o', default='train_result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_data = LocalDepthDataset(args.dataset_path, mode="train")
    test_data = LocalDepthDataset(args.dataset_path, mode="test")

    input_channel = train_data.get_input_channel_size();
    input_channel = 4

    ldp_net = LDP_Net(f_size=64, input_channel=input_channel)

    model = LDPNetTrainChain(ldp_net)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    # TODO : Confirm that add_hook is needed.
    optimizer.add_hook(chainer.optimizer.optimizer_hooks.WeightDecay(rate=0.0005))

    train_data = TransformDataset(train_data, LDDTransform(train_data))

    train_iter = chainer.iterators.SerialIterator(train_data, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batch_size,
                                                 repeat=False, shuffle=False)
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(
        extensions.snapshot_object(model.ldp_net, 'snapshot_model.npz'),
        trigger=(args.iteration, "iteration"))

    log_interval = 20, 'iteration'
    plot_interval = 20, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr', 'main/loss']),
        trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.run()

if __name__ == '__main__':
    main()
