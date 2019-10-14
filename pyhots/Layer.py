import numpy as np
from pyhots.TimeSurface import TimeSurface
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
        self.radius = surface_dimensions[0] // 2
        self.minimum_events = 2 * self.radius
        if network.total_number_of_events != None:
            self.all_timesurfaces = np.zeros((network.total_number_of_events, polarities, surface_dimensions[0], surface_dimensions[1]))
            self.all_best_ids = np.zeros((network.total_number_of_events))
        else:
            self.all_timesurfaces = []
        self.bases = []
        self.basis_activations = np.zeros(number_of_features, dtype=np.int)
        self.processed_events = 0
        self.passed_events = 0
        self.refused_events = 0
        self.min_corr_score = 0.3

        # Reboot bases if they are useless?
        #define JM_REBOOT_CENTERS true  // reboot unused centers?
        #define JM_REBOOT_SINCE 10000   // number of ev without activation before reboot
        #define JM_ACTI_IMMUNITY 25000
        #define JM_ALPHA_REBOOT 0.2
        self.reboot_bases = True
        self.reboot_at = 10000 # number of events without acti before reboot
        self.reboot_immunity = 25000 # once a base has this much number of acti, it can not be rebooted
        self.reboot_factor = 0.2 # factor applied to the reboot
        self.reboot_layer_total_activity = 0
        self.reboot_base_activity = []

        # Init bases (1 new base maximum per file, means that for 10 bases, we need to open the 10 first files just to init the bases)
        self.wait_for_next_file = False

    def enable_new_base(self):
        self.wait_for_next_file = False

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

            # 1st case, we first need to populate bases
            if len(self.bases) < self.number_of_features:
                if not self.wait_for_next_file:
                    self.bases.append(timesurface.data)
                    print('added new base ' + str(len(self.bases)) + '/' + str(self.number_of_features))
                    self.reboot_base_activity.append(0)
                    self.wait_for_next_file = True
                    return None

            else: # else process
                #self.check_prototypes(timesurface)
                best_prototype_id, corr_score = self._correlate_with_bases(timesurface)
                self.reboot_layer_total_activity += 1
                self.reboot_base_activity[best_prototype_id] = self.reboot_layer_total_activity

                if self.all_timesurfaces != []:
                    self.all_timesurfaces[self.processed_events,:,:,:] = timesurface.data
                    self.all_best_ids[self.processed_events] = best_prototype_id

                event.p = best_prototype_id
                self.passed_events += 1

                # check reboot
                if self.reboot_bases:
                    for idbase in range(self.number_of_features):
                        if (self.reboot_layer_total_activity - self.reboot_base_activity[idbase]) > self.reboot_at: # reboot!
                            self.bases[idbase] += self.reboot_factor * (timesurface.data - self.bases[idbase])
                            self.bases[idbase] = np.clip(self.bases[idbase], a_min = 0, a_max= 1)
                            self.reboot_base_activity[idbase] = self.reboot_layer_total_activity
                            self.basis_activations[idbase] = 0
                            print('Reboot ' + str(idbase))


                return event

        else:
            self.refused_events += 1
            return None

    def reset_memory(self):
        # create memory with padding
        self.timestamp_memory = np.zeros((self.polarities,
                                          self.network.sensor_size[0] + self.radius*2,
                                          self.network.sensor_size[1] + self.radius*2))
        self.timestamp_memory -= self.tau * 3 + 1

    #def check_prototypes(self):

    # def _correlate_with_bases(self, timesurface, method='cosine_similarity'):
    #     corrs = []
    #     for index, basis in enumerate(self.bases):
    #         if method == 'cosine_similarity':
    #             corrs.append(self.cosine_similarity(basis, timesurface.data))
    #     best_index = np.argmax(corrs)
    #     best_corr = corrs[best_index]
    #
    #     if self.network.learning_enabled and best_corr > self.min_corr_score:
    #         self.basis_activations[best_index] += 1
    #         learning_rate = self.learning_rate(self.basis_activations[index])
    #         self.bases[best_index] += learning_rate * best_corr * (timesurface.data - self.bases[best_index])
    #     return best_index, best_corr

    def _correlate_with_bases(self, timesurface, method='cosine_similarity'):
            dists = []
            for index, basis in enumerate(self.bases):
                if method == 'cosine_similarity':
                    dists.append(self.cosine_similarity(basis, timesurface.data))
            best_index = np.argmax(dists)
            best_dist = dists[best_index]
            if self.network.learning_enabled:
                # update best base
                self.basis_activations[best_index] += 1
                learning_rate = self.learning_rate(self.basis_activations[index])
                self.bases[best_index] += learning_rate * best_dist * (timesurface.data - self.bases[best_index])

                # # reboot bases?
                # self.acti_since_last_reboot[best_index] += 1
                # self.events_since_reboot += 1
                # if self.events_since_reboot > self.reboot_every:
                #     self.events_since_reboot = 0 # reset counter
                #     for ida, aslr in enumerate(self.acti_since_last_reboot):
                #         if aslr < self.reboot_threshold:
                #             self.bases[ida] += 0.3 * (self.bases[np.random.randint(16)] - self.bases[np.random.randint(16)])
                #             # self.bases[ida] += np.random.random() * (self.bases[np.random.randint(16)] - self.bases[ida])
                #             self.acti_since_last_reboot[ida] = 0
                #             self.basis_activations[ida] = 0
                #             print('reboot base ' + str(ida))

            return best_index, best_dist

    def learning_rate(self, activations):
        return 1 / (1 + activations)

    def cosine_similarity(self, basis, surface):
        return np.sum(basis*surface) / np.sqrt(np.sum(basis**2) * np.sum(surface**2))
