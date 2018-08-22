#coding: 'utf-8'

"""
LDP_Net
ldp_net_train_chain

created by Kazunari on 2018/08/23 
"""

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F

class LDPNetTrainChain(chainer.Chain):
    """
    Calculate losses for LDP Net and report them.
    """

    def __init__(self, ldp_net):
        super(LDPNetTrainChain, self).__init__()
        with self.init_scope():
            self.ldp_net = ldp_net

    def __call__(self, x, t, mask):

        self.y = self.ldp_net(x)

        # TODO : implement loss function
        #self.loss = self._ldp_net_loss(self.y, t, mask)
        self.loss = 0

        chainer.reporter.report({'train_loss': self.loss}, self)

        return self.loss

def _ldp_net_loss(y, t, mask):
    dtype = t.dtype
    xp = chainer.cuda.cuda.get_array_module(t)

    inv_valid_pixels = xp.array(1, t,dtype) / F.sum(mask.astype(dtype))