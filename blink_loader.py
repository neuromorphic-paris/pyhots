#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:15:18 2019

@author: gregorlenz
"""
import os
import loris


def blink_loader(directory):
    print(directory)

    for path, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('blink-labels'):
                recording_number = path[-1]
                event_file = loris.read_file(path + '/' + recording_number + '.es')

                with open(path+'/'+file) as text_file:
                    lines = text_file.readlines()
                    for line in lines:
                        line = line.split()
                        t = line[0]
                        print(t)
