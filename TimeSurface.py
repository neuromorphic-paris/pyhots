import numpy as np
import ipdb


class TimeSurface:
    def __init__(self, layer, timestamp_context):
        timestamp_data = np.exp(timestamp_context/layer.tau)
        timestamp_data[timestamp_context < (-3*layer.tau)] = 0
        assert timestamp_context.shape == (layer.number_of_features, 2 * layer.radius + 1, 2 * layer.radius + 1)
        self.data = timestamp_data

    def normalize(self):
        self.data = self.data / np.sum(self.data)
