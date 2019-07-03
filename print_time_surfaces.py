#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:15:18 2019

@author: gregorlenz
"""

from blink_loader import blink_loader

data_set_base_path = '/home/gregorlenz/Recordings/face-detection/face-detection-data-set/'
data_set = 'indoor'

blinks = blink_loader(data_set_base_path + data_set)

# %% generate time surfaces

numpy.size(blinks)


