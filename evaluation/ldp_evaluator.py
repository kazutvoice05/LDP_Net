#coding: 'utf-8'

"""
LDP_Net
ldp_evaluator

created by Kazunari on 2018/09/11 
"""

import copy
import numpy as np
import tqdm

import chainer
from chainer import reporter
from chainer.training import extensions

from evaluation.metrics import compute_metrics


class LDPNetEvaluator(extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'val'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target, device=None):
        super(LDPNetEvaluator, self).__init__(
            iterator, target, device=device)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            iter = iterator
        else:
            iter = copy.copy(iterator)

        predictions = None
        ts = None
        for batch in iter:
            img, pred_depth, c_map, t, mask = self.converter(batch, self.device)

            y = target(img, pred_depth, c_map)

            y = chainer.cuda.to_cpu(y.data)
            t = chainer.cuda.to_cpu(t)

            if predictions is None:
                predictions = y
                ts = t
            else:
                predictions = np.concatenate([predictions, y])
                ts = np.concatenate([ts, t])

        test_result = compute_metrics(predictions, ts)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(test_result, target)

        return observation

    @staticmethod
    def batch_evaluation(target, *in_arrays):
        img, pred_depth, c_map, t, mask = in_arrays

        y = target(img, pred_depth, c_map)

        y = chainer.cuda.to_cpu(y.data)
        t = chainer.cuda.to_cpu(t)

        batch_result = compute_metrics(y, t)

        return batch_result
