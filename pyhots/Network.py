#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:00:14 2019

@author: gregorlenz
"""
from pyhots.Layer import Layer
import numpy as np
import matplotlib.pyplot as plt
from pyhots.TimeSurface import TimeSurface
import ipdb


class Network():
    def __init__(self, surface_dimensions_per_layer,
                 number_of_features_per_layer,
                 time_constants_per_layer,
                 sensor_size,
                 learning_enabled=True,
                 plot_evolution=True,
                 reboot_bases=True,
                 merge_polarities=False):
        assert len(surface_dimensions_per_layer)\
                == len(number_of_features_per_layer)\
                == len(time_constants_per_layer)
        assert len(sensor_size) == 2

        self.layers = []
        self.learning_enabled = learning_enabled
        self.plot_evolution = plot_evolution
        self.minimum_events = 5
        self.merge_polarities = merge_polarities
        polarities = 1 if merge_polarities else 2
        self.first_layer_polarities = 1 if merge_polarities else 2
        for l, surface_dimension in enumerate(surface_dimensions_per_layer):
            self.layers.append(Layer(self, l, surface_dimension, polarities,
                                     number_of_features_per_layer[l],
                                     time_constants_per_layer[l],
                                     sensor_size,
                                     reboot_bases,))
            polarities = number_of_features_per_layer[l]
        self.number_of_layers = len(self.layers)
        self.sensor_size = sensor_size
        if self.plot_evolution:
            self.fig, self.axes, self.axisImages = self._prepare_plotting(number_of_features_per_layer[0])
        self.processed_recordings = 0

    def __call__(self, recording, label):
        recording = recording.view(type=np.recarray, dtype=[('t', np.int_), ('x', np.int_), ('y', np.int_), ('p', np.int_)])
        recording = recording.reshape(-1)
        assert max(recording.x) < self.sensor_size[0]
        assert max(recording.y) < self.sensor_size[1]
        assert all(t1 <= t2 for t1, t2 in zip(recording.t, recording.t[1:]))
        [layer.reset_memory() for layer in self.layers]
        if self.merge_polarities: recording.p = np.zeros(len(recording))

        # check if there are all cluster center initialised
        if len(self.layers[0].bases) < self.layers[0].number_of_features:
            # look for random event in recording, create surf and add as base
            self.choose_new_basis_from_recording(recording)
            return

        for event in recording:
            for index, layer in enumerate(self.layers):
                event = layer.process(event)
  
        self.processed_recordings += 1

        if self.plot_evolution:
            for index, axisImage in enumerate(self.axisImages):
                if index < len(self.layers[0].bases):
                    img = np.hstack(self.layers[0].bases[index])
                else:
                    size_feature = self.layers[0].surface_dimensions
                    img = np.zeros((size_feature[0], size_feature[1]*self.first_layer_polarities), dtype = float)
                axisImage.set_data(img)
                learning_rate = self.layers[0].learning_rate(self.layers[0].basis_activations[index])
                n_acti = self.layers[0].basis_activations[index]
                stitle = str(index) + ': A=' + str(n_acti) + '\nlr=' + str(round(learning_rate, 5))
                self.axes[index].title.set_text(stitle)
            self.fig.suptitle(str(self.processed_recordings) + ' recordings, ' + str(self.layers[0].processed_events) + ' events in total')
            plt.show(self.fig)
            self.fig.canvas.draw()

    def _prepare_plotting(self, number_of_features):
        plt.close()
        side_length = int(np.sqrt(number_of_features))
        fig, axes = plt.subplots(side_length, side_length)  # , dpi=80)
        axes = np.reshape(axes, -1)
        fig.suptitle('first layer bank')
        axisImages = []
        size_feature = self.layers[0].surface_dimensions
        image_for_plot = np.zeros((size_feature[0], size_feature[1]*self.first_layer_polarities), dtype = float)
        for index, axis in enumerate(axes):
            axisImages.append(axis.imshow(image_for_plot, vmin=0, vmax=1,
                              cmap = plt.cm.hot, interpolation = 'none', origin = 'upper'))
            axis.axis('off')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(axisImages[0], cax=cbar_ax)
        return fig, axes, axisImages

    def choose_new_basis_from_recording(self, recording):
        select = int(np.random.uniform(1, len(recording)))
        event = recording[select]
        radius = self.layers[0].radius
        mask = (recording.t <= event.t)\
                & (recording.t >= event.t - self.layers[0].tau)\
                & (recording.x >= event.x - radius)\
                & (recording.x < event.x + radius + 1)\
                & (recording.y >= event.y - radius)\
                & (recording.y < event.y + radius + 1)
        dims = self.layers[0].surface_dimensions
        time_window = np.zeros((self.first_layer_polarities, dims[0], dims[1]))
        for e in recording[mask]:
            time_window[e.p, e.x + radius - event.x, e.y + radius - event.y] = e.t
        time_surface = TimeSurface(self.layers[0], time_window)
        self.layers[0].bases.append(time_surface.data)
        self.layers[0].reboot_base_activity.append(0)
        #print('Added new base ' + str(len(self.layers[0].bases)) + '/' + str(self.layers[0].number_of_features))