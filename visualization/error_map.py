# -*- coding: utf-8 -*-

"""
visualization
error_map

Created by seiya on 2018/08/28
Copyright (c) 2018 Seiya Ito. All rights reserved.
"""

import cv2
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import *
from tqdm import tqdm
from scipy.io import loadmat

mat = loadmat('data/eigen15/predictions_depth_vgg.mat')
print(mat.keys())

files = sorted(glob.glob('data/eigen15/error_*.tif'))
for i, f in tqdm(enumerate(files)):
    if any(str(n) in f for n in [300, 411, 470, 780, 1022, 1216, 1411]):
        cv2.imwrite(f.replace('error_', '')[:-4] + '-eigen.tif', mat['depths'][..., i].astype(np.float32))

eigen = sorted(glob.glob('data/test/*-eigen.tif'))
pred = sorted(glob.glob('data/test/*-pred.tif'))
gt = sorted(glob.glob('data/test/*-gt.tif'))

for i, (e, p, g) in enumerate(zip(eigen, pred, gt)):
    # e = cv2.resize(cv2.imread(e, cv2.IMREAD_UNCHANGED), (304, 228), cv2.INTER_NEAREST)
    e = cv2.resize(cv2.imread(e, cv2.IMREAD_UNCHANGED), (320, 240), cv2.INTER_NEAREST)
    e = e[6:-6, 8:-8]

    p = cv2.resize(cv2.imread(p, cv2.IMREAD_UNCHANGED), (304, 228), cv2.INTER_NEAREST)

    g = cv2.resize(cv2.imread(g, cv2.IMREAD_UNCHANGED), (320, 240), cv2.INTER_NEAREST)
    g = g[6:-6, 8:-8]

    p = np.exp(p * 0.45723134)

    eg = np.abs(e - g)
    pg = np.abs(p - g)

    print(np.mean(eg), np.mean(pg), np.mean(pg) / np.mean(eg))

    vmin = np.min([eg.min(), pg.min()])
    vmax = np.max([eg.max(), pg.max()])

    dmax = np.max([e.max(), p.max(), g.max()])
    dmin = np.min([e.min(), p.min(), g.min()])

    fig = plt.figure()
    ii = plt.imshow(eg, interpolation='nearest', vmax=vmax, vmin=vmin, cmap='hot')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig('eigen_error-%d.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    fig.clf()

    fig = plt.figure()
    ii = plt.imshow(pg, interpolation='nearest', vmax=vmax, vmin=vmin, cmap='hot')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig('pred_error-%d.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    fig.clf()

    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                   bottom=False, top=False, left=False, right=False)

    ticks = np.linspace(vmin, vmax, num=5)
    ticks = (ticks[1:] + ticks[:-1]) / 2.0

    cbar = matplotlib.colorbar.ColorbarBase(ax, cmap='hot', orientation='horizontal')
    cbar.set_clim([vmin, vmax])
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=30)
    cbar.draw_all()
    plt.savefig('error_bar-%i.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close()

    fig = plt.figure()
    ii = plt.imshow(e, interpolation='nearest', vmax=e.max(), vmin=e.min())
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig('eigen-%d.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close()

    fig = plt.figure()
    ii = plt.imshow(p, interpolation='nearest', vmax=dmax, vmin=dmin)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig('pred-%d.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close()

    fig = plt.figure()
    ii = plt.imshow(g, interpolation='nearest', vmax=dmax, vmin=dmin)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, top=False, left=False, right=False)
    plt.savefig('gt-%d.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close()

    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                   bottom=False, top=False, left=False, right=False)

    ticks = np.linspace(dmin, dmax, num=5)
    ticks = (ticks[1:] + ticks[:-1]) / 2.0

    cbar = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal')
    cbar.set_clim([dmin, dmax])
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=30)
    cbar.draw_all()
    plt.savefig('depth_bar-%d.pdf' % i, bbox_inches="tight", pad_inches=0.0, transparent=True)
    plt.close()
