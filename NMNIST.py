#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:01:27 2019

@author: gregorlenz
"""
import os
import os.path
from Dataset import Dataset
from utils import check_integrity, download_and_extract_archive


class NMNIST(Dataset):
    """ǸMNIST <https://www.garrickorchard.com/datasets/n-mnist>`_ data set.

    arguments:
        train: choose training or test set
        save_to: location to save files to on disk
        transforms: list of transforms to apply to the data
        download: choose to download data or not
    """
    base_url = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/'
    test_zip = base_url + 'AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1'
    train_zip = base_url + 'AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1'
    test_md5 = '69CA8762B2FE404D9B9BAD1103E97832'
    train_md5 = '20959B8E626244A1B502305A9E6E2031'

    def __init__(self, save_to, train=True, transforms=None, download=False):
        super(NMNIST, self).__init__(save_to, transforms=transforms)

        self.train = train

        if train:
            self.url = self.train_zip
            self.file_md5 = self.train_md5
            self.filename = 'nmnist_train.zip'
        else:
            self.url = self.test_zip
            self.file_md5 = self.test_md5
            self.filename = 'nmnist_test.zip'

        self.data = []
        self.targets = []

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def download(self):
        download_and_extract_archive(self.url, self.location_on_system,
                                     filename=self.filename, md5=self.file_md5)

    def _check_integrity(self):
        root = self.location_on_system
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.file_md5):
            return False
        return True
