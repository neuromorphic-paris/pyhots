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
        timestamp_data = np.exp(timestamp_context/self.tau)
        timestamp_data[timestamp_context < (-3*self.tau)] = 0
        timesurface = timestamp_data
        # TODO improve surface
        #timesurface = TimeSurface(timestamp_data)

        # correlate with bases of this layer
        best_prototype_id, corr_score = self._correlate_with_bases(timesurface)

        # if close to one basis, propagate to next layer
        if corr_score > self.min_corr_score:
            event.p = best_prototype_id
            return event
        else:
            return None

    def _correlate_with_bases(self, timesurface):
        ipdb.set_trace()
        if self.network.learning_enabled:
            # update prototype
        pass

    def _corr2(a,b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        d = np.sqrt((a*a).sum() * (b*b).sum())
        if d == 0.0: # a and b are equal to zero...
            return 0.0
        r = (a*b).sum() / d;
        return r