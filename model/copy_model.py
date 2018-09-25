# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:55:36 2017

@author: kaneko.naoshi
"""

import chainer


# This code is inspired by the code from
# http://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614
def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)

    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]

        if type(child) != type(dst_child):
            continue

        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)

        if isinstance(child, chainer.Link):
            status = 'match'

            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    status = 'param mismatch'
                    break

            if status == 'match':
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    b[1].data = a[1].data

            elif status == 'coerce':
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    b[1].data.flat = a[1].data.flat
                print('Coerce {}'.format(child.name))

            else:
                print('Ignore {} because of {}'.format(child.name, status))
