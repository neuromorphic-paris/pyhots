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
import numpy as np
from numpy.lib import recfunctions as rfn

blinks_transformed = []
for blink in blinks:
    blink.x -= min(blink.x)
    blink.y -= min(blink.y)
    new_events = rfn.structured_to_unstructured(blink)
    new_events = np.delete(new_events, 3, 1)
    blinks_transformed.append(new_events)

#blinks = blinks_transformed
#del(blinks_transformed)

from show_td import show_td_surface
show_td_surface(blinks_transformed[0], frame_length=5000, decay_constant=5000, wait_delay=200, scale=10)
# %%
import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

time_constant = 2e5
events = blinks[0]
# create sparse multidimensional matrix with t,x,y and t as data
matrix = sparse.COO((events[:,2], (events[:,2], events[:,0], events[:,1])))
# take max values along the time axis and convert to normal matrix
surface = np.max(matrix, axis=0).todense()
tsurface = np.exp((surface - b.t[-1]) / time_constant)

plt.figure()
sns.heatmap(tsurface)

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(events[:,0],events[:,1],events[:,2])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# %%



array = np.array([[[0],[1],[10]],
         [[1],[1],[11]],
         [[0],[1],[12]],
         [[1],[0],[13]],
         [[1],[1],[14]]])

test=np.zeros((np.max(events[:,0]) - np.min(events[:,0]), np.max(events[:,1]) - np.min(events[:,1]),
         np.max(events[:,2]) - np.min(events[:,2])))

from scipy.sparse import coo_matrix,csr_matrix
coo = coo_matrix((events[:,2], (events[:,0], events[:,1])))
csr = csr_matrix((events[:,2], (events[:,0], events[:,1])))

import sparse
import timeit

#print(timeit.timeit('surface = np.max(sparse.COO((events[:,2], (events[:,2], events[:,0], events[:,1]))).todense(), axis=0)', number=10, globals=globals()))
print(timeit.timeit('surface = np.max(sparse.COO((events[:,2], (events[:,2], events[:,0], events[:,1]))), axis=0)', number=100, globals=globals()))
sp_coo = sparse.COO((events[:,2], (events[:,2], events[:,0], events[:,1])))
surface = np.max(sp_coo.todense(), axis=0)
surface2 = np.max(sp_coo, axis=0)
sp_coo2 = sparse.COO((events[:,2], (events[:,0], events[:,1])))

tsurface = np.zeros([max(events[:,0])+1, max(events[:,1])+1])
for event in events:
    tsurface[event[0],event[1]] = event[2]

cli_string = """\
tsurface = np.zeros([max(events[:,0])+1, max(events[:,1])+1])
for event in events:
    tsurface[event[0],event[1]] = event[2]"""
print(timeit.timeit(cli_string, number=100, globals=globals()))
#np.exp(b.t/time_constant)


#for each blink in blinks:
#    tsurface = np.zeros([ydim,xdim*num_polarities])


