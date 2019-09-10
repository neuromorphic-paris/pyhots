import numpy as np


class TimeSurface:
    def __init__(self, times):
        assert times.shape == (self.polarity, 2 * self.radius + 1, 2 * self.radius + 1)
        self.times = times

    def normalize(self):
        self.data = self.data / np.sum(self.data)

    def correlate_with(self, timesurface):
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



