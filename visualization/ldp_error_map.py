#coding: 'utf-8'

"""
LDP_Net
ldp_error_map

created by Kazunari on 2018/09/26 
"""

import cv2
import glob
import os
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat

from evaluation.metrics import *

root_dir = "/Users/Kazunari/Desktop/0926_DeskConv10_64"
out_dir = osp.join(root_dir, "error_map")

os.makedirs(osp.join(out_dir, "pred"), exist_ok=True)
os.makedirs(osp.join(out_dir, "laina"), exist_ok=True)

pred = sorted(glob.glob(osp.join(root_dir, "pred", "npy", "*")))
laina = sorted(glob.glob(osp.join(root_dir, "laina", "npy", "*")))
gt = sorted(glob.glob(osp.join(root_dir, "gt", "npy", "*")))


p_res = None
l_res = None
g_res = None
m_res = None
for i, (p, l, g) in enumerate(zip(pred, laina, gt)):
    f_num = p.split("/")[-1].split(".")[0]

    p = np.load(p)
    l = np.load(l)
    g = np.load(g)

    eps = np.finfo(np.float32).eps
    mask = eps <= g

    op = p
    ol = l
    og = g
    p = np.expand_dims(np.expand_dims(p, axis=0), axis=0).reshape((1, 1, 1, -1))
    l = np.expand_dims(np.expand_dims(l, axis=0), axis=0).reshape((1, 1, 1, -1))
    g = np.expand_dims(np.expand_dims(g, axis=0), axis=0).reshape((1, 1, 1, -1))
    mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0).reshape((1, 1, 1, -1))

    #rms_pg = np.sqrt(np.sum(np.power(np.abs(p - g) * mask, 2)) / mask.sum())
    #rms_lg = np.sqrt(np.sum(np.power(np.abs(l - g) * mask, 2)) / mask.sum())

    if p_res is None:
        p_res = p
        l_res = l
        g_res = g
        m_res = mask
    else:
        p_res = np.concatenate([p_res, p], axis=3)
        l_res = np.concatenate([l_res, l], axis=3)
        g_res = np.concatenate([g_res, g], axis=3)
        m_res = np.concatenate([m_res, mask], axis=3)

    #c, r = p.shape
    #p_res.append(np.sum(pg) / (c * r))
    #l_res.append(np.sum(lg) / (c * r))

    """
    vmin = min(pg.min(), lg.min())
    vmax = max(pg.max(), lg.max())

    fig = plt.figure()
    ii = plt.imshow(pg, vmax=vmax, vmin=vmin, cmap='hot')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig(osp.join(out_dir, "pred", f_num + ".pdf"), bbox_inches="tight", pad_inches=0.0, transparent=True)
    fig.clf()
    plt.close()

    fig = plt.figure()
    ii = plt.imshow(lg, vmax=vmax, vmin=vmin, cmap='hot')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig(osp.join(out_dir, "laina", f_num + ".pdf"), bbox_inches="tight", pad_inches=0.0, transparent=True)
    fig.clf()
    plt.close()
    """

p_res = np.array(p_res, dtype=np.float64)
l_res = np.array(l_res, dtype=np.float64)
g_res = np.where(g_res >= np.finfo(np.float64).eps, np.array(g_res, dtype=np.float64), np.finfo(np.float64).eps)
m_res = np.array(m_res, dtype=np.float64)

p_res = threshold_accuracy(p_res, g_res, 1.25, m_res, m_res.sum())
l_res = threshold_accuracy(l_res, g_res, 1.25, m_res, m_res.sum())

print("pred:  " + str(p_res))
print("laina: " + str(l_res))