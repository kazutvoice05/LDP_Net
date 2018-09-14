#coding: 'utf-8'

"""
LDP_Net
test_model

created by Kazunari on 2018/08/23 
"""

import numpy as np
import chainer

from model.ldp_net import LDP_Net
from model.ldp_net_train_chain import LDPNetTrainChain
from dataset.LDD_Transform import LDDTransform
from dataset.Local_Depth_Dataset import LocalDepthDataset

ldd = LocalDepthDataset("/Users/Kazunari/projects/datasets/LocalDepthDataset")
ldd_transform = LDDTransform(ldd)

train_data = chainer.datasets.TransformDataset(ldd, ldd_transform)

rgbd_channel = 4
ldp_net = LDP_Net(rgbd_channel=rgbd_channel)

model = LDPNetTrainChain(ldp_net)

img, pred_depth, c_map, t, mask = train_data.get_example(0)

loss = model(img, pred_depth, c_map, t, mask)

print(loss)