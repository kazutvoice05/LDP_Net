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
    batchsize = t.shape[0]
    xp = chainer.cuda.cuda.get_array_module(t)

    m = mask.reshape(batchsize, -1).astype(t.dtype)
    y_m = y.reshape(batchsize, -1) * m
    t_m = t.reshape(batchsize, -1) * m

    diff = y_m - t_m
    num_valid = F.sum(m, axis=1)

    l2_loss = F.sum(num_valid * F.sum(F.square(diff, axis=1)))

    scale_inv_loss = 0.5 * F.sum(F.square(F.sum(diff, axis=1)))

    depth_loss = (
        (l2_loss - scale_inv_loss)/ 
        F.maximum(F.sum(F.square(num_valid)), xp.array(1, num_valid.dtype)))
    
    m_grad_x = xp.logical_and(
        mask[:, :, :, 1:], mask[:, :, :, -1]).astype(t.dtype)
    m_grad_y = xp.logical_and(
        mask[:, :, 1:, :], mask[:, :, -1, :]).astype(t.dtype)
    
    y_grad_x = (y[:, :, :, 1:] - y[:, :, :, :-1])
    y_grad_y = (y[:, :, 1:, :] - y[:, :, :-1, :])
    
    t_grad_x = (t[:, :, :, 1:] - t[:, :, :, :-1])
    t_grad_y = (t[:, :, 1:, :] - t[:, :, :-1, :])

    grad_loss = (
        F.sum(m_grad_x * F.square(y_grad_x - t_grad_x)) / F.sum(m_grad_x)
        + F.sum(m_grad_y * F.square(y_grad_y - t_grad_y)) / F.sum(m_grad_y))
    
    return depth_loss + grad_loss