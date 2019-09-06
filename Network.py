#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:00:14 2019

@author: gregorlenz
"""
import Layer
import numpy as np


class Network():
    def __init__(self, surface_dimensions_per_layer,
                 number_of_features_per_layer, time_constants_per_layer):
        assert len(surface_dimensions_per_layer)\
                == len(number_of_features_per_layer)\
                == len(time_constants_per_layer)

        self.layers = []
        for l in range(0, len(number_of_features_per_layer)):
            self.layers.add(Layer(surface_dimensions_per_layer[l],
                                  number_of_features_per_layer[l],
                                  time_constants_per_layer[l]))
        self.number_of_layers = len(self.layers)

    def __call__(self, recording):
        # cast to rec array so reading code becomes easier since I can use
        # things like event.t and the like
        recording = recording.view(type=np.recarray, dtype=[('t', np.float_),
                                                            ('x', np.float_),
                                                            ('y', np.float_),
                                                            ('p', np.float_)])
        for event in recording:´
            for layer in self.layers:
                event = layer.process(event)
