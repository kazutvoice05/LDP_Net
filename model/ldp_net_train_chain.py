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
        x = x[0]
        t = t[0]
        mask = mask[0]

        self.y = self.ldp_net(x)

        # TODO : Refine Loss Function (Now, Kaneko's Implimentation of Loss is used.)
        self.loss = self._ldp_net_loss(self.y, t, mask)

        chainer.reporter.report({'loss': self.loss}, self)

        return self.loss

    def _ldp_net_loss(self, y, t, mask):
        dtype = t.dtype
        xp = chainer.cuda.get_array_module(t)

        inv_valid_pixels = xp.array(1, t.dtype) / F.sum(mask.astype(dtype))

        l2_diff = F.where(mask, F.squared_error(y, t), xp.zeros(t.shape, dtype))
        l2_loss = F.scale(F.sum(l2_diff), inv_valid_pixels, axis=0)

        scale_inv_diff = F.where(
            mask, F.absolute_error(y, t), xp.zeros(t.shape, dtype))
        scale_inv_loss = F.scale(
            F.square(F.sum(scale_inv_diff)), F.square(inv_valid_pixels), axis=0)

        scale_inv_weight = xp.array(0.5, dtype)

        depth_loss = l2_loss - F.scale(scale_inv_loss, scale_inv_weight, axis=0)

        kernel_x = xp.array([[[[-1, 0, 1]]]], dtype)
        kernel_y = xp.array([[[[-1],
                               [0],
                               [1]]]], dtype)

        y_grad_x = F.convolution_2d(y, kernel_x)
        y_grad_y = F.convolution_2d(y, kernel_y)

        t_grad_x = F.convolution_2d(t, kernel_x)
        t_grad_y = F.convolution_2d(t, kernel_y)

        kernel_grad_x = xp.array([[[[0.5, 0, 0.5]]]], dtype)
        kernel_grad_y = xp.array([[[[0.5],
                                    [0],
                                    [0.5]]]], dtype)

        mask_grad_x = F.floor(F.convolution_2d(mask.astype(dtype), kernel_grad_x))
        mask_grad_y = F.floor(F.convolution_2d(mask.astype(dtype), kernel_grad_y))

        inv_valid_grads_x = xp.array(1, dtype) / F.sum(mask_grad_x)
        inv_valid_grads_y = xp.array(1, dtype) / F.sum(mask_grad_y)

        grad_diff_x = F.where(
            F.cast(mask_grad_x, bool),
            F.squared_error(y_grad_x, t_grad_x),
            xp.zeros(t_grad_x.shape, dtype))
        grad_diff_y = F.where(
            F.cast(mask_grad_y, bool),
            F.squared_error(y_grad_y, t_grad_y),
            xp.zeros(t_grad_y.shape, dtype))

        grad_loss_x = F.scale(F.sum(grad_diff_x), inv_valid_grads_x, axis=0)
        grad_loss_y = F.scale(F.sum(grad_diff_y), inv_valid_grads_y, axis=0)

        grad_loss = grad_loss_x + grad_loss_y

        return depth_loss + grad_loss