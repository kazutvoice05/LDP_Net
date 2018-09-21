#coding: 'utf-8'

"""
LDP_Net
ldp_net_test

created by Kazunari on 2018/09/16 
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

    os.makedirs(out_dir, exist_ok=True)

    test_data = LocalDepthDataset(args.dataset_path, mode="test")

    ldp_net = LDP_Net(f_size=64,
                      rgbd_channel=4,
                      pretrained_model=args.pretrained_model)

    model = LDPNetTrainChain(ldp_net)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    test_data = TransformDataset(test_data, LDDTransform(test_data))

    test_iter = chainer.iterators.SerialIterator(test_data, args.batch_size,
                                                 shuffle=False, repeat=False)

    evaluator = LDPNetEvaluator(test_iter, model.ldp_net, device=args.gpu)

    result = evaluator()

    print(result)

    with open(osp.join(out_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(result, f)
        print("save metrics.pkl")

if __name__ == "__main__":
    main()