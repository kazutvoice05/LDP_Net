#coding: 'utf-8'

"""
LDP_Net
test_model

created by Kazunari on 2018/08/23 
"""

import numpy as np
import chainer

from model.ldp_net import LDP_Net


input_data = np.asarray(np.random.rand(1, 34, 64, 64), dtype=np.float32)

predictor = LDP_Net(input_channel=input_data.shape[1])

data = predictor(input_data)

print("data max: " + str(max(data)) + "\t" + "data min: " + str(min(data)))

print(data)