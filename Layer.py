# -*- coding: utf-8 -*-
import numpy as np
import TimeSurface from TimeSurface

class Layer():
    def __init__(self, surface_dimensions, number_of_features, time_constant):
        self.surface_dimensions = surface_dimensions
        self.number_of_features = number_of_features
        self.time_constant = time_constant
        self.latest_timestamps = 0

    def process(self, event):
        self.tsmem[event.p, event.x, event.y] = event.t

        tcontext = self.tsmem[:_nPol,_x-self.R:_x+self.R+1, _y-self.R:_y+self.R+1] - _t
        tsdata = np.exp(tcontext/self.tau)
        tsdata[tcontext < (-3*self.tau)] = 0
        tsurf = TimeSurface(self.R, _nPol, tsdata)

        if self.do_smoothing:
            tsurf.smooth(self.smoothing_size)

        if tsurf.most_ev_per_pol >= self.power_needed: # WARNING is ev_needed
            if (self.isInhibMem[_pol,_x,_y] == 1) and self.do_inhib:
            # if False:
                self.isInhibMem[_pol,_x,_y] = 0
                return -1, tsurf
            else:
                best_corr = 0.0
                for b in np.arange(0,self.bankCounter):
                    # # print(corr)
                    corr = tsurf.correlate_with(self.bank[b])
                    # print('corr',corr)
                    if corr > best_corr:
                        best_corr = corr
                        best_id = b

                # hysteresis (high and low)
                if best_corr > self.dist_threshold_high:

                    self.activityCounter[best_id] += 1

                    if self.do_update_proto:
                        lr = 1.0 / (1.0 + self.activityCounter[best_id])
                        updated_proto = update_prototype(self.bank[best_id].data, tsurf.data, lr)
                        self.bank[best_id].data = updated_proto

                    # print('next layer, best id1', best_id)
                elif (best_corr == 0.0) and (self.bankCounter > 0):
                    # print('corr is zero, dismiss')
                    best_id = -1
                elif best_corr <= self.dist_threshold_low:
                    # print('newproto', best_corr)
                    self.bank.append(tsurf)
                    self.activityCounter.append(0)
                    self.t_new_proto.append(_t)
                    # if first_ts[0] == 0: # monitoring
                    #     first_ts[0] = ts[ii]
                    best_id = self.bankCounter
                    print('new proto for bank ' + str(self.id) + ' at time ' + str(_t) + ', with id:' + str(self.bankCounter) + ' progress: ' + str(ii) + '%')

                    self.bankCounter += 1
                else: # the dist is in-between the thresholds, just dismiss it.
                    # print('hysteresis: trashed it.')
                    # print('hysteresis',best_corr)
                    best_id = -1

                q,s,d = np.where(tsurf.data == 0) # pol,x,y
                s = s - self.R
                d = d - self.R
                inhib = np.vstack((q,s,d))
                for aa in inhib.T:
                    self.isInhibMem[aa[0], aa[1] + _x, aa[2] + _y] = 1
                return best_id, tsurf
        else:
            return -1, tsurf
