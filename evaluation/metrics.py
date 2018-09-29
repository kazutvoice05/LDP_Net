#coding: 'utf-8'

"""
LDP_Net
metrics

created by Kazunari on 2018/09/11 
"""

import numpy as np


def absolute_relative_diff(y, t, mask, count):
    eps = np.finfo(np.float32).eps
    y = np.where(t > eps, y, eps)

    diff = np.absolute(y - t) / t
    diff = np.sum((diff * mask)) / count
    return diff


def linear_root_mean_squared_error(y, t, mask, count):
    d = np.power(y - t, 2)
    d = d * mask

    diff = np.sqrt((np.sum(d) / count))
    return diff


def linear_mean_error(y, t, mask, count):
    d = np.abs(y - t)
    d = d * mask

    diff = np.sum(d) / count
    return diff


def log10_error(y, t, mask, count):
    diff = np.absolute(np.log10(y) - np.log10(t))
    diff = diff * mask

    diff = np.sum(diff) / count
    return diff


def threshold_accuracy(y, t, threshold, mask, count):
    thresholded_array = np.asarray(np.maximum(y/t, t/y) < threshold, dtype=np.float32)
    true_count = np.count_nonzero(np.where(mask,
                                           thresholded_array,
                                           np.zeros_like(thresholded_array, dtype=np.float32)))
    return true_count / count


def compute_metrics(pred_depths, true_depths, mask):
    y = np.maximum(pred_depths, np.finfo(np.float32).eps)
    t = np.maximum(true_depths, np.finfo(np.float32).eps)
    count = np.count_nonzero(mask)

    th1 = threshold_accuracy(y, t, 1.25, mask, count)
    th2 = threshold_accuracy(y, t, pow(1.25, 2), mask, count)
    th3 = threshold_accuracy(y, t, pow(1.25, 3), mask, count)
    abs_rel = absolute_relative_diff(y, t, mask, count)
    rmse = linear_root_mean_squared_error(y, t, mask, count)
    log10 = log10_error(y, t, mask, count)

    result = {
        'thresh1': th1, 'thresh2': th2, 'thresh3': th3,
        'abs_rel': abs_rel, 'rmse': rmse, 'log10': log10
    }

    return result