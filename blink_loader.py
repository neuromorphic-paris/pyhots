#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:15:18 2019

@author: gregorlenz
"""
import os
import loris

def blink_loader(data_set_path):
    directory = os.fsencode(data_set_path)
    print(directory)
