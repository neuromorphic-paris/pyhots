import numpy as np
import ipdb


class TimeSurface:
    def __init__(self, layer, timestamp_context, decay='lin'):
        if decay == 'lin':
            timestamp_data = timestamp_context / (3 * layer.tau) + 1
            timestamp_data[timestamp_data < 0] = 0
        elif decay == 'exp':
            timestamp_data = np.exp(timestamp_context/layer.tau)
            timestamp_data[timestamp_context < (-3*layer.tau)] = 0
        assert timestamp_context.shape == (layer.polarities, 2 * layer.radius
                                           + 1, 2 * layer.radius + 1)
        self.data = timestamp_data

    def normalize(self):
        self.data = self.data / np.sum(self.data)

    def number_of_events(self):
        return np.sum(self.data > 0)  # , axis=(1,2))

    def entropy(self):
        nts = self.data + 1e-10
        nts /= nts.sum()
        return -(np.log2(nts)*nts).sum()
