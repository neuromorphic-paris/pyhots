import numpy as np
from TimeSurface import TimeSurface
import ipdb

class Layer:
    def __init__(self, surface_dimensions, number_of_features, time_constant,
                 sensor_size):
        self.surface_dimensions = surface_dimensions
        self.number_of_features = number_of_features
        self.tau = time_constant
        self.timestamp_memory = np.zeros((number_of_features, sensor_size[0], sensor_size[1]))
        self.latest_timestamps = 0
        self.radius = np.divide(surface_dimensions[0]-1, 2)
        self.bases = np.random.rand(number_of_features, surface_dimensions[0], surface_dimensions[1])

    def process(self, event):
        self.timestamp_memory[event.p, event.x, event.y] = event.t

        # create time surface
        timestamp_context = self.timestamp_memory[:event.p,
                                                event.x-self.radius:event.x+self.radius,
                                                event.y-self.radius:event.y+self.radius] - event.t
        timestamp_data = np.exp(timestamp_context/self.tau)
        timestamp_data[timestamp_context < (-3*self.tau)] = 0
        timesurface = timestamp_data
        #timesurface = TimeSurface(timestamp_data)

        # TODO improve surface

        # correlate with bases of this layer
        best_prototype_id, corr_score = self.correlate_with_bases(timesurface)

        # if close to one basis, propagate to next layer

        # otherwise create new basis and stop
