from src.model import simlarity_model as model
from src.util import image as image_util
from src.util import matrix
from .model_implements.mobilenet_v3 import ModelnetV3
from .model_implements.vit_base import VitBase
from .model_implements.bit import BigTransfer
import os
import json
import numpy as np


class Similarity:
    def get_models(self):
        return [
            model.SimilarityModel(name='Mobilenet V3', image_size=224, model_cls=ModelnetV3()),
            model.SimilarityModel(name='Big Transfer (BiT)', image_size=224, model_cls=BigTransfer()),
            model.SimilarityModel(name='Vision Transformer', image_size=224, model_cls=VitBase(),
                                  image_input_type='pil'),
        ]

    def check_similarity(self, img_main, dir_path, model):
        imgs = []
        image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                       os.path.isfile(os.path.join(dir_path, f))]
        print('预处理图片开始')
        for url in image_files:
            if url == "": continue
            # imgs.append(image_util.load_image_url(url, required_size=(model.image_size, model.image_size), image_type=model.image_input_type))
            imgs.append(image_util.load_image_file(url, required_size=(model.image_size, model.image_size),
                                                   image_type=model.image_input_type))
        print("预处理图片结束，开始计算图片列表特征")
        features = model.model_cls.extract_feature(imgs)
        feature0 = model.model_cls.extract_feature([image_util.load_image_file(img_main, required_size=(model.image_size, model.image_size),
                                                   image_type=model.image_input_type)])
        results = []
        lastDist = 0
        lastFile = ""
        for i, v in enumerate(features):
            if i == 0: continue
            dist = matrix.cosine(feature0, v)
            if dist > lastDist:
                lastDist = dist
                lastFile = image_files[i]
                print(f'{i} -- distance: {dist}--last-file: {lastFile}')
            # results.append((imgs[i], f'similarity: {int(dist*100)}%'))
            # original_img = image_util.load_image_url(img_urls[i], required_size=None, image_type='pil')
            # original_img = image_util.load_image_file(image_files[i], required_size=None, image_tytpe='pil')
        results.append((lastFile, f'similarity: {dist}'))
        print("end of results")
        return results
    # 连续图片找出变化大的
    def check_similarity_compute(self, dir_path, model):

        parent_dir = os.path.dirname(dir_path)  # 获取上一级目录路径
        output_file = os.path.join(parent_dir, 'imgDataJson.json')  # 拼接文件路径
        if os.path.exists(output_file):
            print(f"文件 {output_file} 存在")
            with open(output_file, 'r') as f:
                features_list = json.load(f)

            # 将列表中的数据转换回字典对象
            features = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in features_list]
        else:
            print(f"文件 {output_file} 不存在")
            image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                           os.path.isfile(os.path.join(dir_path, f))]
            print('计算图片开始')
            features = model.model_cls.extract_feature_dictV2(image_files)
            # 将字典对象中的 ndarray 对象转换为列表
            features_list = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in features]

            # 写入数据到 JSON 文件
            with open(output_file, 'w') as f:
                json.dump(features_list, f)

            print(f"Feature data saved to file: {output_file}")

        return features