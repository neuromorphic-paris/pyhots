import numpy as np
from pyhots.TimeSurface import TimeSurface
import ipdb


class Layer:
    def __init__(self, network, index, surface_dimensions, polarities,
                 number_of_features, time_constant, sensor_size, reboot_bases):
        self.network = network
        self.index = index
        self.surface_dimensions = surface_dimensions
        self.polarities = polarities
        self.number_of_features = number_of_features
        self.tau = time_constant
        self.radius = surface_dimensions[0] // 2
        self.minimum_events = 2 * self.radius
        self.bases = []
        self.basis_activations = np.zeros(number_of_features, dtype=np.int)
        self.processed_events = 0
        self.passed_events = 0
        self.refused_events = 0
        self.reboot_bases = reboot_bases  # Reboot bases if they are useless
        self.reboot_at = 5000  # number of events without activation before reboot
        self.reboot_immunity = 5000  # once a base has this much number of acti, it can not be rebooted
        self.reboot_factor = 1  # factor applied to the reboot
        self.reboot_base_activity = []

    def process(self, event):
        if event == None:
            return None
        # account for padding
        self.timestamp_memory[event.p, event.x+self.radius, event.y+self.radius] = event.t
        timestamp_window = self.timestamp_memory[:,event.x:event.x+self.surface_dimensions[0],
                                                   event.y:event.y+self.surface_dimensions[1]] - event.t
        timesurface = TimeSurface(self, timestamp_window)
        # TODO improve surface

        # correlate with bases of this layer if enough events
        if timesurface.number_of_events() > self.minimum_events:
            best_prototype_id, corr_score = self._correlate_with_bases(timesurface)
            self.reboot_base_activity[best_prototype_id] = self.passed_events

            event.p = best_prototype_id

            # check reboot
            if self.reboot_bases:
                for idbase in range(self.number_of_features):
                    if (self.passed_events - self.reboot_base_activity[idbase]) > self.reboot_at: # reboot!
                        self.bases[idbase] += self.reboot_factor * (timesurface.data - self.bases[idbase])
                        self.basis_activations[idbase] = 0
                        self.reboot_base_activity[idbase] = self.passed_events
                        print('Reboot ' + str(idbase))
            self.passed_events += 1
            self.processed_events += 1
            return event
        else:
            self.refused_events += 1
            self.processed_events += 1
            return None

    def reset_memory(self):
        # create memory with padding
        self.timestamp_memory = np.zeros((self.polarities,
                                          self.network.sensor_size[0] + self.radius*2,
                                          self.network.sensor_size[1] + self.radius*2))
        self.timestamp_memory -= self.tau * 3 + 1

    def _correlate_with_bases(self, timesurface, method='cosine_similarity'):
            dists = []
            mod = np.sum(timesurface.data**2)
            for index, basis in enumerate(self.bases):
                if method == 'cosine_similarity':
                    dists.append(self.cosine_similarity(basis, timesurface.data, mod))
            best_index = np.argmax(dists)
            best_dist = dists[best_index]
            if self.network.learning_enabled:
                self.basis_activations[best_index] += 1
                learning_rate = self.learning_rate(self.basis_activations[index])
                # study time
                self.bases[best_index] += learning_rate\
                                          * best_dist\
                                          * (timesurface.data - self.bases[best_index])
            return best_index, best_dist

    def learning_rate(self, activations):
        return 1 / (1 + activations)

    def cosine_similarity(self, basis, surface, mod):
        return np.sum(basis*surface) / np.sqrt(np.sum(basis**2) * mod)
