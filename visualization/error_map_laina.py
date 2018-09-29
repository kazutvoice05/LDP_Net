#coding: 'utf-8'

"""
LDP_Net
error_map_laina

created by Kazunari on 2018/09/26 
"""

import cv2
import glob
import os
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat

root_dir = "/Users/Kazunari/projects/datasets/LocalDepthDataset_v2/"
out_dir = "./laina_Error_Maps"

os.makedirs(out_dir, exist_ok=True)

laina = sorted(glob.glob(osp.join(root_dir, "test", "predicted_depths", "*")))
gt = sorted(glob.glob(osp.join(root_dir, "test", "depths", "*")))

for i, (l, g) in enumerate(tqdm(zip(laina, gt))):
    l = np.load(l)

    g = cv2.resize(cv2.imread(g, cv2.IMREAD_UNCHANGED), (320, 240), cv2.INTER_NEAREST)
    g = g / 10000

    lg = np.abs(l - g)

    fig = plt.figure()
    ii = plt.imshow(lg, interpolation='nearest', vmax=lg.max(), vmin=lg.min(), cmap='hot')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig(osp.join(out_dir, str(i+1).zfill(6) + ".png"), bbox_inches="tight", pad_inches=0.0, transparent=True)
    fig.clf()
    plt.close()
