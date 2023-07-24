import tensorflow_hub as hub
import numpy as np 

class ModelnetV3():
    def __init__(self):
        module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5" 
        self.module = hub.load(module_handle)

    def extract_feature(self, imgs):
        print('getting with ModelnetV3...')
        features = []
        for img in imgs:
            features.append(np.squeeze(self.module(img)))
        return features