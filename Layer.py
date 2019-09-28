import numpy as np
from TimeSurface import TimeSurface
import ipdb

class Layer:
    def __init__(self, network, surface_dimensions, polarities,
                 number_of_features, time_constant, learning_rate, sensor_size):
        self.network = network
        self.surface_dimensions = surface_dimensions
        self.polarities = polarities
        self.number_of_features = number_of_features
        self.tau = time_constant
        self.learning_rate = learning_rate
        self.timestamp_memory = np.zeros((polarities, sensor_size[0], sensor_size[1]))
        self.latest_timestamps = 0
        self.radius = int(np.divide(surface_dimensions[0]-1, 2))
        self.bases = []
        for f in range(number_of_features):
            self.bases.append(np.random.rand(self.polarities, surface_dimensions[0], surface_dimensions[1]))
        self.min_corr_score = 0.7

    def process(self, event):
        if event == None:
            return None
        self.timestamp_memory[event.p, event.x, event.y] = event.t
        # create time surface
        timestamp_window = self.timestamp_memory[:,event.x-self.radius:event.x+self.radius+1,
                                                    event.y-self.radius:event.y+self.radius+1] - event.t
        timesurface = TimeSurface(self, timestamp_window)
        # TODO improve surface

        # correlate with bases of this layer
        best_prototype_id, corr_score = self._correlate_with_bases(timesurface)

        # if close to one basis, propagate to next layer
        if corr_score > self.min_corr_score:
            event.p = best_prototype_id
            return event
        else:
            return None

    def _correlate_with_bases(self, timesurface, method='euclidian'):
        if timesurface.number_of_events > self.network.minimum_events:
            best_index = -1
            best_corr = 0
            for index, basis in enumerate(self.bases):
                if method == 'euclidian':
                    corr = np.sqrt(np.sum((basis - timesurface.data)**2))
                if corr > best_corr:
                    best_index = index
                    best_corr = corr

        if self.network.learning_enabled:
            self.bases[best_index] += self.learning_rate * best_corr * (timesurface.data - self.bases[best_index])

        return best_index, best_corr
