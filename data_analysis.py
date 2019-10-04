#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:51:54 2019

@author: gregorlenz
"""

from POKERDVS import POKERDVS
from spike_data_augmentation.datasets.dataloader import Dataloader
from Network import Network

testset = POKERDVS(save_to='./data')

# %%
testloader = Dataloader(testset, shuffle=False)

# %%
# build surfaces for each recording


# compute k clusters using normalised dot product


# for each k, calculate the total within-cluster sum of square (wss)


# plot wss against number of clusters
