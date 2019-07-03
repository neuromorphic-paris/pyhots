import numpy as np

class TimeSurface:
    def __init__(self, _R, _nPol, _data):
        self.R = _R     # half width
        self.P = _nPol  # number of pol
        if _data.shape == (self.P, 2 * self.R + 1, 2 * self.R + 1):
            self.data = _data
        else:
            print('TS Error: Wrong shape!')

        self.events_per_pol = np.zeros((self.P), dtype = float)
        for p in np.arange(self.P):
            self.events_per_pol[p] = np.sum(self.data[p,:,:] > 0)
        maxevents = (2 * self.R + 1) ** 2
        self.maxpower = np.max(self.events_per_pol / maxevents)
        self.total_ev = np.sum(self.events_per_pol)
        self.most_ev_per_pol = np.max(self.events_per_pol).astype(int)
        self.smoothed = False

    def correlate_with(self, _TS, _type = 'simple'):
        if _TS.P > self.P:
            Pmax = _TS.P
            diff = _TS.P - self.P
            pad = np.zeros((diff, 2 * self.R + 1, 2 * self.R + 1), dtype = float)
            TS_padded = np.vstack((self.data, pad))
            TS_sum = TS_padded + _TS.data # sum both TS

            corr_per_pol = np.zeros((Pmax), dtype = float)
            for p in np.arange(0, Pmax):
                corr_per_pol[p] = corr2(TS_padded[p,:,:],_TS.data[p,:,:])
        elif _TS.P < self.P:
            Pmax = self.P
            diff = self.P - _TS.P
            pad = np.zeros((diff, 2 * self.R + 1, 2 * self.R + 1), dtype = float)
            TS_padded = np.vstack((_TS.data, pad))
            TS_sum = TS_padded + self.data # sum both TS
            # compute correlation for each polarity.
            corr_per_pol = np.zeros((Pmax), dtype = float)
            for p in np.arange(0, Pmax):
                corr_per_pol[p] = corr2(TS_padded[p,:,:],self.data[p,:,:])
        else:
            TS_sum = self.data + _TS.data
            Pmax = self.P
            # compute correlation for each polarity.
            corr_per_pol = np.zeros((self.P), dtype = float)
            for p in np.arange(0, Pmax):
                corr_per_pol[p] = corr2(self.data[p,:,:],_TS.data[p,:,:])
        # compute density per pol.
        density_per_pol = np.zeros((Pmax), dtype = float)
        for p in np.arange(0, Pmax):
            density_per_pol[p] = np.sum(TS_sum[p,:,:])
        density_per_pol = density_per_pol / np.sum(density_per_pol)
        corr = np.dot(corr_per_pol, density_per_pol)
        return corr

    def normalize(self):
        self.data = self.data / np.sum(self.data)

    def smooth(self, _size):
        if not self.smoothed:
            weights = np.full(_size, 1.0/(_size[0]*_size[1]))
            for p in np.arange(self.P):
                self.data[p,:,:] = convolve_scipy(self.data[p,:,:], weights)
            self.smoothed = True
        else:
            print('TS Error: TS already smoothed.')
