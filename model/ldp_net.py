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

    def __init__(self, f_size=64, rgbd_channel=4, n_class=30, class_channel=3):
        self.f_size = f_size
        self.rgbd_channel = rgbd_channel
        self.n_class = n_class
        self.class_channel = class_channel
        self.main_input_channel = rgbd_channel + self.class_channel

        initializer = chainer.initializers.Normal()

        super(LDP_Net, self).__init__()
        with self.init_scope():
            self.conv0_1 = L.Convolution2D(rgbd_channel, rgbd_channel, 1, initialW=initializer)
            self.conv0_2 = L.Convolution2D(self.n_class, self.class_channel, 1, initialW=initializer)
            self.conv0_3 = L.Convolution2D(self.main_input_channel, self.main_input_channel, 1, initialW=initializer)

            self.conv1_1 = L.Convolution2D(self.main_input_channel, 64, 3, pad=1, initialW=initializer)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv1_3 = L.DilatedConvolution2D(64, 64, 3, pad=2, dilate=2, initialW=initializer)
            self.conv1_4 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.bn1_1 = L.BatchNormalization(64)
            self.bn1_2 = L.BatchNormalization(64)
            self.bn1_3 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv2_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv2_3 = L.DilatedConvolution2D(64, 64, 5, pad=4, dilate=2, initialW=initializer)
            self.conv2_4 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.bn2_1 = L.BatchNormalization(64)
            self.bn2_2 = L.BatchNormalization(64)
            self.bn2_3 = L.BatchNormalization(64)

            self.conv3_1 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv3_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv3_3 = L.DilatedConvolution2D(64, 64, 5, pad=4, dilate=2, initialW=initializer)
            self.conv3_4 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.bn3_1 = L.BatchNormalization(64)
            self.bn3_2 = L.BatchNormalization(64)
            self.bn3_3 = L.BatchNormalization(64)

            self.conv4_1 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv4_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv4_3 = L.DilatedConvolution2D(64, 64, 5, pad=4, dilate=2, initialW=initializer)
            self.conv4_4 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.bn4_1 = L.BatchNormalization(64)
            self.bn4_2 = L.BatchNormalization(64)
            self.bn4_3 = L.BatchNormalization(64)

            self.conv5_1 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv5_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv5_3 = L.DilatedConvolution2D(64, 64, 5, pad=4, dilate=2, initialW=initializer)
            self.conv5_4 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.bn5_1 = L.BatchNormalization(64)
            self.bn5_2 = L.BatchNormalization(64)
            self.bn5_3 = L.BatchNormalization(64)

            self.conv6_1 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv6_2 = L.Convolution2D(64, 1, 3, pad=1, initialW=initializer)

    def __call__(self, x_1, x_2):

        """
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        d_h = self.bn1_3(self.conv1_3(h))
        h = F.relu(F.add[h, d_h])

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        d_h = self.bn2_3(self.conv2_3(h))
        h = F.relu(F.add[h, d_h])

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        d_h = self.bn3_3(self.conv3_3(h))
        h = F.relu(F.add[h, d_h])

        h = F.relu(self.bn4_1(self.conv4_1(h)))
        h = F.relu(self.bn4_2(self.conv4_2(h)))
        d_h = self.bn4_3(self.conv4_3(h))
        h = F.relu(F.add[h, d_h])

        h = F.relu(self.bn5_1(self.conv5_1(h)))
        h = F.relu(self.bn5_2(self.conv5_2(h)))
        d_h = self.bn5_3(self.conv5_3(h))
        h = F.relu(F.add[h, d_h])
        """

        h = self.conv0_1(x_1)
        d_h = self.conv0_2(x_2)
        h = F.relu(self.conv0_3(F.concat([h, d_h])))

        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        d_h = self.conv1_3(h)
        h = F.relu(self.conv1_4(F.concat([h, d_h])))

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        d_h = self.conv2_3(h)
        h = F.relu(self.conv2_4(F.concat([h, d_h])))

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        d_h = self.conv3_3(h)
        h = F.relu(self.conv3_4(F.concat([h, d_h])))

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        d_h = self.conv4_3(h)
        h = F.relu(self.conv4_4(F.concat([h, d_h])))

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        d_h = self.conv5_3(h)
        h = F.relu(self.conv5_4(F.concat([h, d_h])))

        h = self.conv6_1(h)
        pred = self.conv6_2(h)

        return pred
