#coding: 'utf-8'

"""
LDP_Net
metrics

created by Kazunari on 2018/09/11 
"""

import numpy as np


def absolute_relative_diff(y, t):
    eps = np.finfo(np.float32).eps
    y = np.where(t > eps, y, eps)

    diff = np.sum(np.absolute(y - t) / t) / np.where(t > eps)[0].size
    return diff


def linear_root_mean_squared_error(y, t):
    d = y - t
    diff = np.sqrt(np.mean(d * d))
    return diff


def log_root_mean_squared_error(y, t):
    d = y - t
    diff = np.sqrt(np.mean(d * d))
    return diff


def log10_error(y, t):
    diff = np.mean(np.absolute(np.log10(y) - np.log10(t)))
    return diff


def threshold_accuracy(y, t, threshold):
    count = np.where(np.maximum(y / t, t / y) < threshold)[0].size
    return count / float(t.size)


def compute_metrics(pred_depths, true_depths):
    y = np.maximum(pred_depths, np.finfo(np.float32).eps)
    t = np.maximum(true_depths, np.finfo(np.float32).eps)

    th1 = threshold_accuracy(y, t, 1.25)
    th2 = threshold_accuracy(y, t, pow(1.25, 2))
    th3 = threshold_accuracy(y, t, pow(1.25, 3))
    abs_rel = absolute_relative_diff(y, t)
    rmse = linear_root_mean_squared_error(y, t)
    log10 = log10_error(y, t)

    result = {
        'thresh1': th1, 'thresh2': th2, 'thresh3': th3,
        'abs_rel': abs_rel, 'rmse': rmse, 'log10': log10
    }

    return result