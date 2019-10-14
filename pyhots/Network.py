#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:00:14 2019

@author: gregorlenz
"""
from pyhots.Layer import Layer
import numpy as np
import matplotlib.pyplot as plt
import sparse
import ipdb


class Network():
    def __init__(self, surface_dimensions_per_layer,
                 number_of_features_per_layer,
                 time_constants_per_layer,
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
                                     sensor_size))
            polarities = number_of_features_per_layer[l]
        self.number_of_layers = len(self.layers)
        self.sensor_size = sensor_size
        if self.plot_evolution:
            self.fig, self.axes, self.axisImages = self._prepare_plotting(number_of_features_per_layer[0])
        self.processed_recordings = 0

    def __call__(self, recording):
        # recording = recording.view(type=np.recarray, dtype=[('t', np.float_), ('x', np.float_),
                                                     # ('y', np.float_), ('p', np.float_)])
        recording['is_increase'] = recording['is_increase'].astype(np.int8)
        recording = recording.view(type=np.recarray,
                                   dtype=[('t', '<u8'), ('x', '<u2'),
                                          ('y', '<u2'), ('p', np.int8)])
        assert max(recording.x) < self.sensor_size[0]
        assert max(recording.y) < self.sensor_size[1]
        assert all(t1 <= t2 for t1, t2 in zip(recording.t, recording.t[1:]))
        [layer.reset_memory() for layer in self.layers]

        if len(self.layers[0].bases) < self.layers[0].number_of_features:
            # look for random event in recording, create surf and add as base
            select = int(np.random.uniform(1, len(recording)))
            event = recording[select]
            radius = self.layers[0].radius

            ipdb.set_trace()
            mask = (recording.t <= event.t)\
                    & (recording.x > event.x - radius)\
                    & (recording.x < event.x + radius + 1)\
                    & (recording.y > event.y - radius)\
                    & (recording.y < event.y + radius + 1)
            #surface = np.zeros(self.layers[0].surface_dimensions)
            surface = np.max(sparse.COO((recording[mask].t, (recording[mask].t, recording[mask].x, recording[mask].y))), axis=0)
            print('added new base ' + str(len(self.bases)) + '/' + str(self.number_of_features))

        for event in recording:
            for index, layer in enumerate(self.layers):
                event = layer.process(event)

        self.processed_recordings += 1

        if self.plot_evolution:
            for index, axisImage in enumerate(self.axisImages):
                if index < len(self.layers[0].bases):
                    img = np.hstack((self.layers[0].bases[index][0],self.layers[0].bases[index][1]))
                else:
                    size_feature = self.layers[0].surface_dimensions
                    img = np.zeros((size_feature[0], size_feature[1]*2), dtype = float)
                axisImage.set_data(img)
                learning_rate = self.layers[0].learning_rate(self.layers[0].basis_activations[index])
                n_acti = self.layers[0].basis_activations[index]
                stitle = 'A=' + str(n_acti) + '\nlr=' + str(round(learning_rate, 5))
                self.axes[index].title.set_text(stitle)

            self.fig.suptitle(str(self.processed_recordings) + ' processed recordings')
            plt.pause(0.01)

    def _prepare_plotting(self, number_of_features):
        plt.close()
        side_length = int(np.sqrt(number_of_features))
        fig, axes = plt.subplots(side_length, side_length)  # , dpi=80)
        axes = np.reshape(axes, -1)
        fig.suptitle('first layer bank')
        axisImages = []
        size_feature = self.layers[0].surface_dimensions
        image_for_plot = np.zeros((size_feature[0], size_feature[1]*2), dtype = float)
        for index, axis in enumerate(axes):
            axisImages.append(axis.imshow(image_for_plot, vmin=0, vmax=1,
                              cmap = plt.cm.hot, interpolation = 'none', origin = 'upper'))
            axis.axis('off')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(axisImages[0], cax=cbar_ax)
        plt.pause(0.0001)
        # figManager = plt.get_current_fig_manager()
        # figManager.window.setGeometry(0, 0, 900, 1500)
        # figManager.window.setFocus()
        return fig, axes, axisImages