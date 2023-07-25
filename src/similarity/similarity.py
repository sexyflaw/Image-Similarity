from src.model import simlarity_model as model
from src.util import image as image_util
from src.util import matrix
from .model_implements.mobilenet_v3 import ModelnetV3
from .model_implements.vit_base import VitBase
from .model_implements.bit import BigTransfer
import os


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
        results.append((lastFile, f'similarity: {int(dist * 100)}%'))
        print("end of results")
        return results
