import numpy as np
from TimeSurface import TimeSurface
import ipdb


class Layer:
    def __init__(self, network, index, surface_dimensions, polarities,
                 number_of_features, time_constant, sensor_size):
        self.network = network
        self.index = index
        self.surface_dimensions = surface_dimensions
        self.polarities = polarities
        self.number_of_features = number_of_features
        self.tau = time_constant
        self.radius = int(np.divide(surface_dimensions[0]-1, 2))
        self.minimum_events = 8
        if network.total_number_of_events != None:
            self.all_timesurfaces = np.zeros((network.total_number_of_events, polarities, surface_dimensions[0], surface_dimensions[1]))
            self.all_best_ids = np.zeros((network.total_number_of_events))
        else:
            self.all_timesurfaces = []
        self.bases = []
        for f in range(number_of_features):
            self.bases.append(np.random.rand(self.polarities, surface_dimensions[0], surface_dimensions[1]))
        self.basis_activations = np.zeros(number_of_features, dtype=np.int)
        self.processed_events = 0
        self.passed_events = 0
        self.refused_events = 0
        self.min_corr_score = 0.3

    def process(self, event):
        if event == None:
            return None
        self.processed_events += 1

        # account for padding
        self.timestamp_memory[event.p, event.x+self.radius, event.y+self.radius] = event.t
        timestamp_window = self.timestamp_memory[:,event.x:event.x+self.surface_dimensions[0],
                                                   event.y:event.y+self.surface_dimensions[1]] - event.t
        timesurface = TimeSurface(self, timestamp_window)
        # TODO improve surface

        # correlate with bases of this layer if enough events
        if timesurface.number_of_events() > self.minimum_events:
            best_prototype_id, corr_score = self._correlate_with_bases(timesurface)

            if self.all_timesurfaces != []:
                self.all_timesurfaces[self.processed_events,:,:,:] = timesurface.data
                self.all_best_ids[self.processed_events] = best_prototype_id

            # if close to one basis, propagate to next layer
            if corr_score < 0:
                print(corr_score)
            if corr_score > self.min_corr_score:
                event.p = best_prototype_id
                self.passed_events += 1
                return event
            else:
                return None
        else:
            self.refused_events += 1
            return None

    def reset_memory(self):
        # create memory with padding
        self.timestamp_memory = np.zeros((self.polarities,
                                          self.network.sensor_size[0] + self.radius*2,
                                          self.network.sensor_size[1] + self.radius*2))
        self.timestamp_memory -= self.tau * 3 + 1

    def _correlate_with_bases(self, timesurface, method='euclidian'):
        corrs = []
        for index, basis in enumerate(self.bases):
            if method == 'euclidian':
                corrs.append(self._correlation(basis, timesurface.data))
        best_index = np.argmax(corrs)
        best_corr = corrs[best_index]

        if self.network.learning_enabled and best_corr > self.min_corr_score:
            self.basis_activations[best_index] += 1
            learning_rate = self.learning_rate(self.basis_activations[index])
            self.bases[best_index] += learning_rate * (timesurface.data - self.bases[best_index])
        return best_index, best_corr

    def learning_rate(self, activations):
        # return 0.1 / (1 + activations / 1000)
        return 1 / (1 + activations / 100)

    def _correlation(self, a, b):
        return (a*b).sum() / (np.sqrt((a*a).sum() * (b*b).sum()))
