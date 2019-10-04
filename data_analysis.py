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
import ipdb
import sparse

testset = POKERDVS(save_to='./data')

# %%
testloader = Dataloader(testset, shuffle=False)

# %%
surface_dimensions = [5, 5]
radius = int(surface_dimensions[0]/2)
number_of_features = [16]
time_constants = [1e4]
learning_rates = [0.075, 0.0012]
sensor_size = (35, 35)
polarities = 2
# pre-allocate time surface space
all_surfaces = np.zeros((testset.total_number_of_events(), 2, surface_dimensions[0], surface_dimensions[1]))
timestamp_memory = np.zeros((polarities, sensor_size[0] + radius*2, sensor_size[1] + radius*2))
timestamp_memory -= time_constants[0] * 3 + 1
# build surfaces for each recording
#for events, label in iter(testloader):
events, label = next(iter(testloader))

events = events.view(dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('p', np.int8)])

for i, event in enumerate(events):
    timestamp_memory[event.p, event.x+radius, event.y+radius] = event.t
    timestamp_window = timestamp_memory[:,event.x:event.x+surface_dimensions[0],
                                               event.y:event.y+surface_dimensions[1]] - event.t
    timestamp_data = np.exp(timestamp_window/time_constants[0])
    timestamp_data[timestamp_window < (-3*time_constants[0])] = 0
    if i > 100:
        ipdb.set_trace()

mask = (events.x >= events.x - radius) & (events.x < events.x + radius)\
       & (events.y >= events.y - radius) & (events.y < events.y + radius)
surface = np.max(sparse.COO((events.t, (events.t, events.x, events.y))).todense(), axis=0)

# compute k clusters using normalised dot product


# for each k, calculate the total within-cluster sum of square (wss)


# plot wss against number of clusters
