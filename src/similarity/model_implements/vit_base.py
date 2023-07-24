from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import numpy as np 
import torch

class VitBase():

    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    def extract_feature(self, imgs):
        features = []
        for img in imgs:
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states =  outputs.last_hidden_state            
            features.append(np.squeeze(last_hidden_states.numpy()).flatten())            
        return features
