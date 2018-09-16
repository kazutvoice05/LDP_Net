#coding: 'utf-8'

"""
LDP_Net
extractor

created by Kazunari on 2018/09/15 
"""

import chainer
import chainer.functions as F
import chainer.links as L


class Extractor(chainer.Chain):

    def __init__(self, extract=False,
                 pretrained_model=None):

        self.extract = extract
        initializer = chainer.initializers.Normal()

        super(Extractor, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, pad=3, initialW=initializer)
            self.bn1 = L.BatchNormalization(64)

            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1, initialW=initializer)
            self.conv1_3 = L.Convolution2D(64, 128, 3, pad=1, initialW=initializer)
            self.conv1_4 = L.Convolution2D(128, 128, 3, pad=1, initialW=initializer)
            self.conv1_5 = L.Convolution2D(128, 256, 3, pad=1, initialW=initializer)
            self.conv1_6 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv1_7 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv1_8 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv1_9 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv1_10 = L.Convolution2D(256, 64, 3, pad=1, initialW=initializer)

            self.bn1_2 = L.BatchNormalization(64)
            self.bn1_3 = L.BatchNormalization(128)
            self.bn1_4 = L.BatchNormalization(128)
            self.bn1_5 = L.BatchNormalization(256)
            self.bn1_6 = L.BatchNormalization(256)
            self.bn1_7 = L.BatchNormalization(256)
            self.bn1_8 = L.BatchNormalization(256)
            self.bn1_9 = L.BatchNormalization(256)
            self.bn1_10 = L.BatchNormalization(64)

            self.conv2_2 = L.Convolution2D(64, 128, 5, stride=2, pad=2, initialW=initializer)
            self.conv2_3 = L.Convolution2D(128, 128, 3, pad=1, initialW=initializer)
            self.conv2_4 = L.Convolution2D(128, 128, 3, pad=1, initialW=initializer)
            self.conv2_5 = L.Convolution2D(128, 256, 5, stride=2, pad=2, initialW=initializer)
            self.conv2_6 = L.Convolution2D(256, 256, 3, pad=1, initialW=initializer)
            self.conv2_7 = L.Convolution2D(256, 512, 5, pad=2, initialW=initializer)
            self.conv2_8 = L.Convolution2D(512, 512, 3, pad=1, initialW=initializer)
            self.conv2_9 = L.Convolution2D(512, 512, 3, pad=1, initialW=initializer)
            self.conv2_10 = L.Convolution2D(512, 64, 3, pad=1, initialW=initializer)

            self.bn2_2 = L.BatchNormalization(128)
            self.bn2_3 = L.BatchNormalization(128)
            self.bn2_4 = L.BatchNormalization(128)
            self.bn2_5 = L.BatchNormalization(256)
            self.bn2_6 = L.BatchNormalization(256)
            self.bn2_7 = L.BatchNormalization(512)
            self.bn2_8 = L.BatchNormalization(512)
            self.bn2_9 = L.BatchNormalization(512)
            self.bn2_10 = L.BatchNormalization(64)

            self.conv3_1 = L.Convolution2D(128, 128, 1, initialW=initializer)
            self.conv3_2 = L.Convolution2D(128, 64, 3, pad=1, initialW=initializer)
            self.conv3_3 = L.Convolution2D(64, 1, 3, pad=1, initialW=initializer)

            self.bn3_1 = L.BatchNormalization(128)
            self.bn3_2 = L.BatchNormalization(64)


        if pretrained_model is not None:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, img, c_map):
        h = F.relu(self.bn1(self.conv1(img)))

        h_1 = F.relu(self.bn1_2(self.conv1_2(h)))
        h_1 = F.relu(self.bn1_3(self.conv1_3(h_1)))
        h_1 = F.relu(self.bn1_4(self.conv1_4(h_1)))
        h_1 = F.relu(self.bn1_5(self.conv1_5(h_1)))
        h_1 = F.relu(self.bn1_6(self.conv1_6(h_1)))
        h_1 = F.relu(self.bn1_7(self.conv1_7(h_1)))
        h_1 = F.relu(self.bn1_8(self.conv1_8(h_1)))
        h_1 = F.relu(self.bn1_9(self.conv1_9(h_1)))
        h_1 = F.relu(self.bn1_10(self.conv1_10(h_1)))

        h_2 = F.relu(self.bn2_2(self.conv2_2(h)))
        h_2 = F.relu(self.bn2_3(self.conv2_3(h_2)))
        h_2 = F.relu(self.bn2_4(self.conv2_4(h_2)))
        h_2 = F.relu(self.bn2_5(self.conv2_5(h_2)))
        h_2 = F.relu(self.bn2_6(self.conv2_6(h_2)))
        h_2 = F.relu(self.bn2_7(self.conv2_7(h_2)))
        h_2 = F.relu(self.bn2_8(self.conv2_8(h_2)))
        h_2 = F.relu(self.bn2_9(self.conv2_9(h_2)))
        h_2 = F.relu(self.bn2_10(self.conv2_10(h_2)))

        h_2 = F.unpooling_2d(h_2, 3, stride=2)
        h_2 = F.unpooling_2d(h_2, 3, stride=2)

        h = F.concat([h_1, h_2])

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))

        if self.extract:
            return h

        pred = self.conv3_3(h)

        return pred
