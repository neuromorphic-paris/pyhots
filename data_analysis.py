#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:51:54 2019

@author: gregorlenz
"""

from POKERDVS import POKERDVS
from spike_data_augmentation.datasets.dataloader import Dataloader
from TimeSurface import TimeSurface
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
import sparse

testset = POKERDVS(save_to='./data')

# %%
testloader = Dataloader(testset, shuffle=False)

# %%
surface_dimensions = [11, 11]
radius = int(surface_dimensions[0]/2)
number_of_features = [16]
time_constants = [1e3]
learning_rates = [0.075, 0.0012]
sensor_size = (35, 35)
polarities = 2
number_of_events = testset.total_number_of_events()
# pre-allocate time surface space
all_surfaces = np.zeros((number_of_events, 2, surface_dimensions[0], surface_dimensions[1]))

# build surfaces for each recording

i = 0
for recording, label in iter(testloader):
    if label == 'di':
        recording = recording.view(dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('p', np.int8)])
        timestamp_memory = np.zeros((polarities, sensor_size[0] + radius*2, sensor_size[1] + radius*2))
        timestamp_memory -= time_constants[0] * 3 + 1
        for event in recording:
            timestamp_memory[event.p, event.x+radius, event.y+radius] = event.t
            timestamp_window = timestamp_memory[:,event.x:event.x+surface_dimensions[0],
                                                       event.y:event.y+surface_dimensions[1]] - event.t
            timestamp_data = np.exp(timestamp_window/time_constants[0])
            timestamp_data[timestamp_window < (-3*time_constants[0])] = 0
            all_surfaces[i,:,:,:] = timestamp_data
            if i > 89850:
                #ipdb.set_trace()
                pass
            i += 1
        break
all_surfaces = all_surfaces[:i, :, :, :]
# print things
from mpl_toolkits.mplot3d import Axes3D
plt.close()
fig = plt.figure()
#ax = fig.gca(projection='3d')

surface = all_surfaces[-118][0]
x = np.arange(0,surface_dimensions[0])
y = x
x,y = np.meshgrid(x, y)
#surf = ax.plot_surface(x, y, surface, cmap=plt.cm.coolwarm,
 #                      linewidth=0, antialiased=False)
sns.heatmap(surface)

# compute k clusters using normalised dot product


# for each k, calculate the total within-cluster sum of square (wss)


# plot wss against number of clusters
