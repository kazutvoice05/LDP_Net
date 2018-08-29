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

    model = LDP_Net(pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_id(args.gpu).use()
        model.to_gpu()

    errors = []

    for i in range(1,20000,50):
        x_1, x_2, t, mask = dataset.get_example(i)

        x_1 = np.expand_dims(x_1, axis=0)
        x_2 = np.expand_dims(x_2, axis=0)

        pred = model(x_1, x_2)

        np_pred = np.asarray(pred.data[0], dtype=np.float32)

        print("max: " + str(np.max(t)))
        
        """
        elems_error = np.abs(np.diff([np_pred, t], axis=0))

        valid_pixel = np.count_nonzero(mask)

        masked_error = np.where(mask, elems_error, np.zeros_like(elems_error, dtype=np.float32))

        ldp_error = np.sum(masked_error) / valid_pixel
        
        #Plot result
        fig = plt.figure()
        ii = plt.imshow(pred.data[0, 0, :, :], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        """
        

if __name__ == '__main__':
    main()