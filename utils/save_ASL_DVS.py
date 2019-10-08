#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:42:04 2019

@author: gregorlenz
"""

import numpy as np
import scipy.io as sio
import loris

mat = sio.loadmat("/home/gregorlenz/Téléchargements/a/a_0004.mat")

x = mat['x'].reshape(-1)
y = mat['y'].reshape(-1)
ts = mat['ts'].reshape(-1)
pol = mat['pol'].reshape(-1)


test = np.rec.fromarrays([ts, x, y, pol], names='t,x,y,is_increase',
                         dtype=[('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('is_increase', '?')])

dvs_file_path = '/home/gregorlenz/Nextcloud/Rest/examples/dvs.es'
dvs_file = loris.read_file(dvs_file_path)

width = int(max(x)+1)
height = int(max(y)+1)
mat_file = {'type': 'dvs', 'width': width, 'height':height, 'events':test}
loris.write_events_to_file(mat_file, '/home/gregorlenz/Téléchargements/test.es')