#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:15:18 2019

@author: gregorlenz
"""
import os
import loris
import numpy as np

blink_length = 300000


def blink_loader(directory):
    print(directory)
    blinks = []
    for path, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('blink-labels'):
                recording_number = path[-1]
                event_file = loris.read_file(path + '/' + recording_number
                                             + '.es')
                events = event_file['events']

                with open(path+'/'+file) as text_file:
                    lines = text_file.readlines()
                    for line in lines:
                        t, xlt, ylt, xlb, ylb, xrt, yrt, xrb, yrb = [int(v) for v in line.split()]

                        ts = events['t']
                        xs = events['x']
                        ys = events['y']

                        blink_events_left = (ts > t) & (ts < t + blink_length)\
                            & (xs > xlt) & (xs < xlb)\
                            & (ys < ylt) & (ys > ylb)\
                            & (~events['is_threshold_crossing'])

                        blink_events_right = (ts > t) & (ts < t + blink_length)\
                            & (xs > xrt) & (xs < xrb)\
                            & (ys < yrt) & (ys > yrb)\
                            & (~events['is_threshold_crossing'])

                        blinks.append(events[blink_events_left])
                        blinks.append(events[blink_events_right])

    return blinks
