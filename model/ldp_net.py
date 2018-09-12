#coding: 'utf-8'

"""
LDP_Net
ldp_net

created by Kazunari on 2018/08/22 
"""

import chainer
import chainer.functions as F
import chainer.links as L



class LDP_Net(chainer.Chain):

    def __init__(self,
                 f_size=64,
                 rgbd_channel=4,
                 n_class=30,
                 class_channel=3,
                 pretrained_model=None):
        self.f_size = f_size
        self.rgbd_channel = rgbd_channel
        self.n_class = n_class
        self.class_channel = class_channel
        self.main_input_channel = rgbd_channel + self.class_channel

        initializer = chainer.initializers.Normal()

        super(LDP_Net, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(self.rgbd_channel, 64, 3, pad=1, initialW=initializer)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv1_3 = L.Convolution2D(64, 128, 3, pad=1, initialW=initializer)
            self.conv1_4 = L.Convolution2D(128, 128, 3, pad=1, initialW=initializer)
            self.conv1_5 = L.Convolution2D(128, 256, 3, pad=1, initialW=initializer)
            self.conv1_6 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv1_7 = L.Convolution2D(256, 512, 3, pad=1, initialW=initializer)
            self.conv1_8 = L.Convolution2D(512, 512, 3, pad=1, initialW=initializer)
            self.conv1_9 = L.Convolution2D(512, 64, 3, pad=1, initialW=initializer)
            self.conv1_10 = L.Convolution2D(64, 1, 3, pad=1, initialW=initializer)

            self.bn1_1 = L.BatchNormalization(64)
            self.bn1_2 = L.BatchNormalization(64)
            self.bn1_3 = L.BatchNormalization(128)
            self.bn1_4 = L.BatchNormalization(128)
            self.bn1_5 = L.BatchNormalization(256)
            self.bn1_6 = L.BatchNormalization(256)
            self.bn1_7 = L.BatchNormalization(512)
            self.bn1_8 = L.BatchNormalization(512)
            self.bn1_9 = L.BatchNormalization(64)

        if pretrained_model is not None:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, img, depth, c_map):
        x = F.concat([img, depth])
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        h = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn1_5(self.conv1_5(h)))
        h = F.relu(self.bn1_6(self.conv1_6(h)))
        h = F.relu(self.bn1_7(self.conv1_7(h)))
        h = F.relu(self.bn1_8(self.conv1_8(h)))
        h = F.relu(self.bn1_9(self.conv1_9(h)))
        pred = self.conv1_10(h)

        return pred
