import tensorflow_hub as hub
import numpy as np 

class BigTransfer:

    def __init__(self):
        self.module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1")

    def extract_feature(self, imgs):
        features = []
        for img in imgs:
            features.append(np.squeeze(self.module(img)))
        return features