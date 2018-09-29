#coding: 'utf-8'

"""
LDP_Net
ldp_resnet

created by Kazunari on 2018/09/21 
"""

from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from chainercv.links.model.resnet import ResNet50
from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet.resblock import ResBlock
from chainercv.links import PickableSequentialChain
from chainercv import utils

from .copy_model import copy_model

class UpBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels):

        initialW = initializers.Normal(0.1)

        super(UpBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, 5, pad=2, initialW=initialW)
            self.conv2 = L.Convolution2D(out_channels, out_channels, 3, pad=1, initialW=initialW)
            self.bn1 = L.BatchNormalization(out_channels)
            self.bn2 = L.BatchNormalization(out_channels)

            self.proj = L.Convolution2D(in_channels, out_channels, 5, pad=2, initialW=initialW)

    def __call__(self, x, **kwargs):
        x = self._up_sampling(x)

        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))

        proj = self.proj(x)

        out = F.relu(F.add(h, proj))

        return out

    def _up_sampling(self, x):
        _, indices = F.max_pooling_2d(x, ksize=1, return_indices=True)
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        x = F.upsampling_2d(x, indices, 2, 2, outsize=outsize)

        return x


class LDP_ResNet(PickableSequentialChain):
    _blocks = [3, 4, 6]
    sunrgbd_mean = np.array([125.91209, 116.54863, 110.43808],
                            dtype=np.float32)[:, np.newaxis, np.newaxis]

    #image_net_model_path = "/home/takagi.kazunari/projects/LDP_Net/model/pre_trained_models/resnet50_imagenet_converted_2018_03_07.npz"
    image_net_model_path = "/Users/Kazunari/projects/taurus/LDP_Net/model/pre_trained_models/resnet50_imagenet_converted_2018_03_07.npz"

    def __init__(self, n_class = None,
                 pretrained_model=None,
                 mean=None, fc_kwargs={}):
        stride_first = True
        conv1_no_bias = False

        blocks = self._blocks

        self.mean = self.sunrgbd_mean
        self.pretrained_model = pretrained_model

        initialW = initializers.HeNormal(scale=1, fan_option='fan_out')
        if self.pretrained_model:
            initialW = initializers.constant.Zero()
            fc_kwargs['initialW'] = initializers.constant.Zero()
        kwargs = {'initialW': initialW, 'stride_first': stride_first}

        super(LDP_ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 1, 3, nobias=conv1_no_bias, initialW=initialW)
            self.res2 = ResBlock(blocks[0], None, 64, 256, 1, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 128, 512, 2, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 256, 1024, 2, **kwargs)

            self.conv5 = Conv2DBNActiv(1024, 256, 1, initialW=initialW, activ=None)

            self.up_conv1 = UpBlock(256, 128)
            self.up_conv2 = UpBlock(128, 64)

            self.conv6 = Conv2DBNActiv(65, 65, 5, 1, 2, initialW=initialW)
            self.conv7 = Conv2DBNActiv(65, 65, 5, 1, 2, initialW=initialW)

            self.pred = L.Convolution2D(65, 1, 5, pad=2, initialW=initialW)

    def __call__(self, img, pred_depth, c_map, **kwargs):
        h = self.conv1(img)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)

        h = self.conv5(h)

        h = self.up_conv1(h)
        h = self.up_conv2(h)

        h = F.concat([h, pred_depth])

        h = self.conv6(h)
        h = self.conv7(h)

        pred = self.pred(h)

        return pred