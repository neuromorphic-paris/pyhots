# -*- coding: utf-8 -*-
import os
import os.path
import loris
from spike_data_augmentation.datasets.dataset import Dataset


class POKERDVS(Dataset):
    classes = ["cl", "he", "di", "sp"]

    def __init__(self, file_dir, save_to='./data', transform=None):
        super(POKERDVS, self).__init__(save_to, transform=transform)

        counts = {'cl': 0, 'he': 0, 'di': 0, 'sp': 0}
        for path, dirs, files in os.walk(file_dir):
            files.sort()
            for file in files:
                if file.endswith('dat'):
                    label = file[:2]
                    if counts[label] < 16:
                        counts[label] += 1
                        event_file = loris.read_file(path + '/' + file)
                        events = event_file['events']
                        events['y'] = 239 - events['y']
                        self.data.append(events)
                        self.targets.append(label)

    def __getitem__(self, index):
        events, target = self.data[index], self.targets[index]

        if self.transform is not None:
            events = self.transform(events)

        return events, target

    def __len__(self):
        return len(self.data)
