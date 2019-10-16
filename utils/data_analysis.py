#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:51:54 2019

@author: gregorlenz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import ipdb

# %%
surfs = first.all_timesurfaces.copy()
# delete all empty surfaces
surfs = surfs[surfs.sum(axis=(1,2,3)) != 0]
# reshape into 2D array (flattening surfaces)
surfs = surfs.reshape(-1,11*11)

kmeans = KMeans(n_clusters=16).fit(surfs)
centers = kmeans.labels_.reshape(-1,11,11)

# %%
fig, axes = plt.subplots(4, 4)
axes = np.reshape(axes, -1)

for index, axis in enumerate(axes):
    axis.imshow(centers[index], vmin=0, vmax=1,
                cmap = plt.cm.hot, interpolation = 'none', origin = 'upper')

