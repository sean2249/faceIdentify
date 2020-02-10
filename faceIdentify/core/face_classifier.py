# coding: utf-8
import numpy as np

class FaceClassifier:
    def __init__(self, features, is_single_user=None):
        self.model = features 

    def identify(self, feature):
        dist = np.linalg.norm(self.model-feature, axis=1)
        return np.mean(dist)

    def recognize(self, feature):
        pass