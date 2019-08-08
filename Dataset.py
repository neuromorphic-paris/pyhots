# -*- coding: utf-8 -*-

import pathlib


class Dataset():
    def __init__(self, save_to=str(pathlib.Path.home()), transforms=None):
        self.location_on_system = save_to
        self.transforms = transforms

    def __repr__(self):
        return "Dataset " + self.__class__.__name__
