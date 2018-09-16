#coding: 'utf-8'

"""
LDP_Net
ldp_net

created by Kazunari on 2018/08/22 
"""

import chainer
import chainer.functions as F
import chainer.links as L

from .extractor import Extractor


class LDP_Net(chainer.Chain):

    def __init__(self, extractor_model,
                 pretrained_model=None):

        self.extractor = Extractor(extractor_model, extract=True)
        del self.extractor.conv1_10

        initializer = chainer.initializers.HeNormal()

        super(LDP_Net, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(65, 65, 5, pad=2, nobias=True, initialW=initializer)
            self.conv2 = L.Convolution2D(65, 65, 5, pad=2, nobias=True, initialW=initializer)
            self.conv3 = L.Convolution2D(65, 1, 5, pad=2, nobias=True, initialW=initializer)
            self.bn1 = L.BatchNormalization(65)
            self.bn2 = L.BatchNormalization(65)

        if pretrained_model is not None:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, img, depth, c_map):
        x = self.extractor(img, c_map, extract=True)

        x = F.concat([x, depth])

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        pred = self.conv3(h)

        return pred


