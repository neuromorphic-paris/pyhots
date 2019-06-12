# -*- coding: utf-8 -*-


class Layer:
    def __init__(self, id, tau, neighbourhood, number_of_features):
        self.id = id
        self.tau = tau
        self.neighbourhood = neighbourhood
        self.number_of_features = number_of_features
