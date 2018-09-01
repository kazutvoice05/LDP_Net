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
                 rgb_channel=3,
                 n_class=30,
                 class_channel=3,
                 pretrained_model=None):
        self.f_size = f_size
        self.rgb_channel = rgb_channel
        self.n_class = n_class
        self.class_channel = class_channel
        self.main_input_channel = rgb_channel + self.class_channel

        initializer = chainer.initializers.Normal()

        super(LDP_Net, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(self.rgb_channel, 64, 3, pad=1, initialW=initializer)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv1_3 = L.Convolution2D(64, 128, 5, pad=2, initialW=initializer)
            self.bn1_1 = L.BatchNormalization(64)
            self.bn1_2 = L.BatchNormalization(64)
            self.bn1_3 = L.BatchNormalization(128)

            self.conv2_1 = L.Convolution2D(128, 128, 3, pad=1, initialW=initializer)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1, initialW=initializer)
            self.conv2_3 = L.Convolution2D(128, 256, 5, pad=2, initialW=initializer)
            self.bn2_1 = L.BatchNormalization(128)
            self.bn2_2 = L.BatchNormalization(128)
            self.bn2_3 = L.BatchNormalization(256)

            self.conv3_1 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv3_3 = L.Convolution2D(256, 512, 5, pad=2, initialW=initializer)
            self.bn3_1 = L.BatchNormalization(256)
            self.bn3_2 = L.BatchNormalization(256)
            self.bn3_3 = L.BatchNormalization(512)

            self.conv4_1 = L.Convolution2D(512, 512, 3, pad=1, initialW=initializer)
            self.conv4_2 = L.Convolution2D(512, 512, 3, pad=1, initialW=initializer)
            self.conv4_3 = L.Convolution2D(512, 512, 5, pad=2, initialW=initializer)
            self.bn4_1 = L.BatchNormalization(512)
            self.bn4_2 = L.BatchNormalization(512)
            self.bn4_3 = L.BatchNormalization(512)

            self.conv_c1 = L.Convolution2D(self.n_class, 3, 1, initialW=initializer)
            self.conv_c2 = L.Convolution2D(3, 64, 1, initialW=initializer)
            self.conv_c3 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.bn_c1 = L.BatchNormalization(3)
            self.bn_c2 = L.BatchNormalization(64)
            self.bn_c3 = L.BatchNormalization(64)

            self.conv5_1 = L.Convolution2D(576, 576, 1, initialW=initializer)
            self.conv5_2 = L.Convolution2D(576, 128, 3, pad=1, initialW=initializer)
            self.conv5_3 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.bn5_1 = L.BatchNormalization(576)
            self.bn5_2 = L.BatchNormalization(128)
            self.bn5_3 = L.BatchNormalization(64)

            self.conv6_1 = L.Convolution2D(65, 65, 5, pad=2, initialW=initializer)
            self.conv6_2 = L.Convolution2D(65, 1, 5, pad=2, initialW=initializer)
            self.bn6_1 = L.BatchNormalization(65)

        if pretrained_model is not None:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, img, depth, c_map):

        h = F.relu(self.bn1_1(self.conv1_1(img)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.relu(self.bn2_3(self.conv2_3(h)))

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))

        h = F.relu(self.bn4_1(self.conv4_1(h)))
        h = F.relu(self.bn4_2(self.conv4_2(h)))
        h = F.relu(self.bn4_3(self.conv4_3(h)))

        c_h = F.relu(self.bn_c1(self.conv_c1(c_map)))
        c_h = F.relu(self.bn_c2(self.conv_c2(c_h)))
        c_h = F.relu(self.bn_c3(self.conv_c3(c_h)))

        f_h = F.concat([h, c_h])

        f_h = F.relu(self.bn5_1(self.conv5_1(f_h)))
        f_h = F.relu(self.bn5_2(self.conv5_2(f_h)))
        f_h = F.relu(self.bn5_3(self.conv5_3(f_h)))

        f_h = F.concat([f_h, depth])

        f_h = F.relu(self.bn6_1(self.conv6_1(f_h)))
        pred = self.conv6_2(f_h)

        return pred
