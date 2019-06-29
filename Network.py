#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:00:14 2019

@author: gregorlenz
"""
import Layer


class Network():
    def __init__(self, surface_dimensions_per_layer,
                 number_of_features_per_layer, time_constants_per_layer):
        # check for equal lengths of input vectors
        self.number_of_layers = len(number_of_features_per_layer)
        self.layers = []
        for l in range(1, self.number_of_layers):
            self.layers.add(Layer(surface_dimensions_per_layer[l],
                                  number_of_features_per_layer[l],
                                  time_constants_per_layer[l]))
