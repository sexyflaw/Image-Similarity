import os.path
import tensorflow_hub as hub
import numpy as np
from src.util import image as image_util


class ModelnetV3():
    def __init__(self):
        # module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5"
        module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
        self.module = hub.load(module_handle)

    def extract_feature(self, imgs):
        print('getting with ModelnetV3...')
        features = []
        for img in imgs:
            features.append(np.squeeze(self.module(img)))
        return features

    def extract_feature_dict(self, imgs):
        print('getting with ModelnetV3...')
        features = []
        for img in imgs:
            filename = os.path.basename(img.filename)
            feature_dict = {"filename": filename, "feature": np.squeeze(self.module(img))}
            features.append(feature_dict)
        return features

    def extract_feature_dictV2(self, image_files):
        features = []
        for url in image_files:
            if url == "": continue
            # imgs.append(image_util.load_image_url(url, required_size=(model.image_size, model.image_size), image_type=model.image_input_type))
            # imgs.append(image_util.load_image_file(url, required_size=(model.image_size, model.image_size), image_type=model.image_input_type))
            img = image_util.load_image_file(url, image_type='array')
            filename = os.path.basename(url)
            feature_dict = {"filename": filename, "feature": np.squeeze(self.module(img))}
            features.append(feature_dict)

        return features

    def process_frames(self, frames, frame_indices):
        # 批量处理多个帧
        features = []
        for frame, frame_index in zip(frames, frame_indices):
            img = image_util.load_frame(frame, image_type='array')
            feature_dict = {"filename": frame_index, "feature": np.squeeze(self.module(img))}
            features.append(feature_dict)
        return features
