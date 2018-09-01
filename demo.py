#coding: 'utf-8'

"""
LDP_Net
demo.py

created by Kazunari on 2018/08/29 
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import chainer

import sys
import os.path as osp
sys.path.append(osp.curdir)
from model.ldp_net import LDP_Net
from dataset.Local_Depth_Dataset import LocalDepthDataset
from dataset.LDD_Transform import LDDTransform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--pretrained_model', '-m')
    parser.add_argument('--dataset_path', '-p', default="/Users/Kazunari/projects/datasets/LocalDepthDataset")
    parser.add_argument('--out', '-o', default=None)

    args = parser.parse_args()

    ldd = LocalDepthDataset(args.dataset_path)
    dataset = chainer.datasets.TransformDataset(ldd, LDDTransform(ldd))

    model = LDP_Net(rgbd_channel=3, pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    errors = []

    for i in range(1, 20000, 50):
        x_1, x_2, t, mask = dataset.get_example(i)

        x_1 = np.expand_dims(x_1, axis=0)[:, :3, :, :]
        x_2 = np.expand_dims(x_2, axis=0)

        pred = model(x_1, x_2)

        np_pred = np.asarray(pred.data[0], dtype=np.float32)

        elems_error = np.abs(np.diff([np_pred, t], axis=0))
        valid_pixel = np.count_nonzero(mask)
        masked_error = np.where(mask, elems_error, np.zeros_like(elems_error, dtype=np.float32))
        ldp_error = np.sum(masked_error) / valid_pixel

        elems_error = np.abs(np.diff([x_1[:, 3, :, :] / 10, t], axis=0))
        masked_error = np.where(mask, elems_error, np.zeros_like(elems_error, dtype=np.float32))
        eigen_error = np.sum(masked_error) / valid_pixel

        print("ldp_n error: " + str(ldp_error))
        print("eigen error: " + str(eigen_error))

if __name__ == '__main__':
    main()