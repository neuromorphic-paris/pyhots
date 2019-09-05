# -*- coding: utf-8 -*-
import os
import os.path
import loris
from spike_data_augmentation import Dataset


class POKERDVS(Dataset):
    def __init__(self, save_to, train=True, transform=None, download=False):
        super(POKERDVS, self).__init__(save_to, transform=transform)

        for path, dirs, files in os.walk('/home/gregorlenz/Development/Github/HOTS-DOJO/Datasets/Cards/usable/pips'):
            files.sort()
            for file in files:
                if file.endswith('dat'):
                    event_file = loris.read_file(path + '/' + file)
                    events = event_file['events']
                    self.data.append(events)
                    self.targets.append(file[:2])

    def __getitem__(self, index):
        events, target = self.data[index], self.targets[index]

        if self.transform is not None:
            events = self.transform(events)

        return events, target

    def __len__(self):
        return len(self.data)
