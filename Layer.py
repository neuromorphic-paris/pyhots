import numpy as np
from TimeSurface import TimeSurface
import ipdb

class Layer:
    def __init__(self, network, surface_dimensions, number_of_features, time_constant,
                 sensor_size):
        self.surface_dimensions = surface_dimensions
        self.number_of_features = number_of_features
        self.tau = time_constant
        self.timestamp_memory = np.zeros((number_of_features, sensor_size[0], sensor_size[1]))
        self.latest_timestamps = 0
        self.radius = int(np.divide(surface_dimensions[0]-1, 2))
        self.bases = np.random.rand(number_of_features, surface_dimensions[0], surface_dimensions[1])
        self.min_corr_score = 0.7
        self.network = network

    def process(self, event):
        if event == None:
            return None
        self.timestamp_memory[event.p, event.x, event.y] = event.t
        # create time surface
        timestamp_context = self.timestamp_memory[:,event.x-self.radius:event.x+self.radius+1,
                                                    event.y-self.radius:event.y+self.radius+1] - event.t
        timesurface = TimeSurface(self, timestamp_context)
        # TODO improve surface

        # correlate with bases of this layer
        best_prototype_id, corr_score = self._correlate_with_bases(timesurface)

        # if close to one basis, propagate to next layer
        if corr_score > self.min_corr_score:
            event.p = best_prototype_id
            return event
        else:
            return None

    def _correlate_with_bases(self, timesurface):
        corr_per_pol = np.zeros((self.number_of_features))
        for index, basis in enumerate(self.bases):
            ipdb.set_trace()
            corr_per_pol[index] = self._corr2(basis, timesurface.data)

        if timesurface > self.network.minimum_events:
            pass
        if self.network.learning_enabled:
            # update prototype
            pass

    def _corr2(self, a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        d = np.sqrt((a*a).sum() * (b*b).sum())
        if d == 0.0: # a and b are equal to zero...
            return 0.0
        r = (a*b).sum() / d;
        return r