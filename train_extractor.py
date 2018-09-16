#coding: 'utf-8'

"""
LDP_Net
train_extractor

created by Kazunari on 2018/09/15 
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
from chainerui.extensions import CommandsExtension

chainer.set_debug(True)

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from model.extractor import Extractor
from model.extractor_train_chain import ExtractorTrainChain
from dataset.Local_Depth_Dataset import LocalDepthDataset
from dataset.LDD_Transform import LDDTransform
from evaluation.extractor_evaluator import ExtractorEvaluator

def main():
    parser = argparse.ArgumentParser(
        description="Training script for LDP Net"
    )
    parser.add_argument('--dataset_path', '-p', type=str,
                        default="/home/takagi.kazunari/projects/datasets/LocalDepthDataset_v2")
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--multi_gpu', action="store_true")
    parser.add_argument('--pretrained_model', '-m', default=None)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--out', '-o', default='extractor_train_result',
                        help='Root directory of output')
    parser.add_argument('--dir', '-d', default=None,
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=50000)
    parser.add_argument('--normalize_depth', '-n', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.out is not "extractor_train_result":
        out_dir = args.out
    elif args.out is "extractor_train_result" and args.dir is not None:
        out_dir = osp.join(args.out, args.dir)
    else:
        import datetime
        out_dir = osp.join(args.out, "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now()))

    train_data = LocalDepthDataset(args.dataset_path, mode="train")
    test_data = LocalDepthDataset(args.dataset_path, mode="test")

    extractor = Extractor()

    model = ExtractorTrainChain(extractor)

    if args.multi_gpu:
        import chainermn

        # Setting for Multi GPU Training
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank

        n_node = comm.intra_rank
        n_gpu = comm.size
        chainer.cuda.get_device_from_id(device).use()

        total_batch_size = n_gpu * args.batch_size

        args.lr = args.lr * total_batch_size

        optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9), comm)
    elif args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

        args.lr = args.lr * args.batch_size
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    else:
        args.lr = args.lr * args.batch_size
        optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)


    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.optimizer_hooks.WeightDecay(rate=0.0005))

    train_data = TransformDataset(train_data, LDDTransform(train_data, normalize=args.normalize_depth))
    test_data = TransformDataset(test_data, LDDTransform(test_data, normalize=args.normalize_depth))

    if args.multi_gpu:
        if comm.rank != 0:
            train_data = None
            test_data = None
        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batch_size,
                                                 shuffle=False, repeat=False)

    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (args.iteration, 'iteration'), out=out_dir)

    trainer.extend(
        extensions.snapshot_object(model.extractor, 'model_{.updater.iteration}.npz'),
        trigger=(150, "iteration"))

    log_interval = 5, 'iteration'
    print_interval = 5, 'iteration'
    evaluation_interval = 150, 'iteration'

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss', 'main/depth_loss', 'main/grad_loss', 'main/triplet_loss',
         'main/accuracy_gain', 'val/main/abs_rel', 'val/main/rmse']),
        trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=5))

    trainer.extend(ExtractorEvaluator(test_iter,
                                   model.extractor,
                                   device=args.gpu),
                   trigger=evaluation_interval)

    trainer.extend(CommandsExtension())

    trainer.run()

if __name__ == '__main__':
    main()
