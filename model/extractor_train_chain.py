#coding: 'utf-8'

"""
LDP_Net
extractor_train_chain

created by Kazunari on 2018/09/15 
"""

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F

class ExtractorTrainChain(chainer.Chain):
    """
    Calculate losses for LDP Net and report them.
    """

    def __init__(self, extractor):
        super(ExtractorTrainChain, self).__init__()
        with self.init_scope():
            self.extractor = extractor

    def __call__(self, img, pred_depth, c_map, t, mask):

        self.y = self.extractor(img, c_map)

        # TODO : Refine Loss Function (Now, Kaneko's Implimentation of Loss is used.)
        depth_loss, grad_loss, triplet_loss = self.ldp_net_loss(self.y, pred_depth, t, mask)

        self.loss = depth_loss + grad_loss + triplet_loss

        y_e, b_e = self.rmse(self.y, pred_depth, t, mask)

        chainer.reporter.report({'loss': self.loss,
                                 'depth_loss': depth_loss,
                                 'grad_loss': grad_loss,
                                 'triplet_loss': triplet_loss,
                                 'accuracy_gain': b_e - y_e,
                                 'LDP_rmse': y_e,
                                 'Eigen_rmse': b_e}, self)

        return self.loss

    def ldp_net_loss(self, y, pred_depth, t, mask):
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

        # Avoid zero dividing when all elements in mask are False.
        sum_mask_grad_x = F.sum(mask_grad_x)
        sum_mask_grad_y = F.sum(mask_grad_y)
        sum_mask_grad_x.data = max(xp.asarray([1], dtype=xp.float32), sum_mask_grad_x.data)
        sum_mask_grad_y.data = max(xp.asarray([1], dtype=xp.float32), sum_mask_grad_y.data)

        inv_valid_grads_x = xp.array(1, dtype) / sum_mask_grad_x
        inv_valid_grads_y = xp.array(1, dtype) / sum_mask_grad_y

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

        masked_y = F.where(mask, y, xp.zeros(t.shape, dtype))[0, 0]
        masked_pred_depth = F.where(mask, pred_depth, xp.zeros(t.shape, dtype))[0, 0]

        triplet_loss = F.triplet(masked_y, t[0, 0], masked_pred_depth) / 100
        triplet_loss = 0

        return depth_loss, grad_loss, triplet_loss

    def rmse(self, y, b, t, mask):
        y = np.asarray(cuda.to_cpu(y.data), dtype=np.float32)
        b = np.asarray(cuda.to_cpu(b), dtype=np.float32)
        t = np.asarray(cuda.to_cpu(t), dtype=np.float32)
        mask = np.asarray(cuda.to_cpu(mask), dtype=np.float32)
        y_e = []
        b_e = []

        for i in range(y.shape[0]):
            y_elem_error = np.abs(np.diff([y[i, :, :, :], t[i, :, :, :]], axis=0))
            b_elem_error = np.abs(np.diff([b[i, :, :, :], t[i, :, :, :]], axis=0))

            valid_pixel = np.count_nonzero(mask)

            y_masked_error = np.where(mask, y_elem_error, np.zeros_like(y_elem_error, dtype=np.float32))
            b_masked_error = np.where(mask, b_elem_error, np.zeros_like(b_elem_error, dtype=np.float32))

            y_elem_error = np.sum(y_masked_error) / valid_pixel
            b_elem_error = np.sum(b_masked_error) / valid_pixel

            y_e.append(y_elem_error)
            b_e.append(b_elem_error)

        y_e = np.asarray(y_e, dtype=np.float32)
        b_e = np.asarray(b_e, dtype=np.float32)

        y_e = np.sum(y_e) / y.shape[0]
        b_e = np.sum(b_e) / b.shape[0]

        return y_e, b_e
