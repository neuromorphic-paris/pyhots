#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:00:14 2019

@author: gregorlenz
"""
from Layer import Layer
import numpy as np
import matplotlib.pyplot as plt
import ipdb


class Network():
    def __init__(self, surface_dimensions_per_layer,
                 number_of_features_per_layer,
                 time_constants_per_layer,
                 learning_rates_per_layer,
                 sensor_size,
                 learning_enabled=True,
                 plot_evolution=True,
                 total_number_of_events=None):
        assert len(surface_dimensions_per_layer)\
                == len(number_of_features_per_layer)\
                == len(time_constants_per_layer)
        assert len(sensor_size) == 2

        self.layers = []
        self.learning_enabled = learning_enabled
        self.plot_evolution = plot_evolution
        self.minimum_events = 5
        self.total_number_of_events = total_number_of_events
        polarities = 2  # On and Off events in the first layer
        for l, surface_dimension in enumerate(surface_dimensions_per_layer):
            self.layers.append(Layer(self, l, surface_dimension, polarities,
                                     number_of_features_per_layer[l],
                                     time_constants_per_layer[l],
                                     learning_rates_per_layer[l],
                                     sensor_size))
            polarities = number_of_features_per_layer[l]
        self.number_of_layers = len(self.layers)
        self.sensor_size = sensor_size
        if self.plot_evolution:
            self.fig, self.axisImages = self._prepare_plotting(number_of_features_per_layer[0])
        self.processed_recordings = 0

    def __call__(self, recording):
        # cast to rec array so reading code becomes easier since I can use
        # things like event.t and the like
        # recording = recording.view(type=np.recarray, dtype=[('t', np.float_), ('x', np.float_),
                                                     # ('y', np.float_), ('p', np.float_)])
        recording['is_increase'] = recording['is_increase'].astype(np.int8)
        recording = recording.view(type=np.recarray,
                                   dtype=[('t', '<u8'), ('x', '<u2'),
                                          ('y', '<u2'), ('p', np.int8)])
        assert max(recording.x) < self.sensor_size[0]
        assert max(recording.y) < self.sensor_size[1]
        assert all(x <= y for x, y in zip(recording.t, recording.t[1:]))
        [layer.reset_memory() for layer in self.layers]
        for event in recording:
            for index, layer in enumerate(self.layers):
                event = layer.process(event)
                #if index == 0:
                #feature_number = event.p
                #self.ax[feature_number].set_data(layer.bases[feature_number][0])
                #self.fig.suptitle(str(layer.processed_events) + ' processed events in layer ' + str(index))

        self.processed_recordings += 1

        if self.plot_evolution:
            for index, axisImage in enumerate(self.axisImages):
                axisImage.set_data(self.layers[0].bases[index][0])
            self.fig.suptitle(str(self.processed_recordings) + ' processed recordings')
            plt.pause(0.01)

    def _prepare_plotting(self, number_of_features):
        plt.close()
        side_length = int(np.sqrt(number_of_features))
        fig, axes = plt.subplots(side_length,side_length)
        axes = np.reshape(axes, -1)
        fig.suptitle('first layer bases')
        ax = []
        for index, axis in enumerate(axes):
            #vmin = 0, vmax = 1,
            ax.append(axis.imshow(self.layers[0].bases[index][0], cmap = plt.cm.hot, interpolation = 'none', origin = 'upper'))
            axis.axis('off')
        plt.pause(0.0001)
        return fig, ax
