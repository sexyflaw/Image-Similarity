from src.model import simlarity_model as model 
from src.util import image as image_util
from src.util import matrix
from .model_implements.mobilenet_v3 import ModelnetV3
from .model_implements.vit_base import VitBase
from .model_implements.bit import BigTransfer


class Similarity:
    def get_models(self):
        return [
            model.SimilarityModel(name= 'Mobilenet V3', image_size= 224, model_cls = ModelnetV3()),
            model.SimilarityModel(name= 'Big Transfer (BiT)', image_size= 224, model_cls = BigTransfer()),
            model.SimilarityModel(name= 'Vision Transformer', image_size= 224, model_cls = VitBase(), image_input_type='pil'),
            ]        

    def check_similarity(self, img_urls, model):
        imgs = []
        for url in img_urls:
            if url == "": continue
            #imgs.append(image_util.load_image_url(url, required_size=(model.image_size, model.image_size), image_type=model.image_input_type))
            imgs.append(image_util.load_image_file(url, required_size=(model.image_size, model.image_size),
                                                  image_type=model.image_input_type))
        
        features = model.model_cls.extract_feature(imgs)
        results = []
        for i, v in enumerate(features):
            if i == 0: continue 
            dist = matrix.cosine(features[0], v)
            print(f'{i} -- distance: {dist}')
            # results.append((imgs[i], f'similarity: {int(dist*100)}%'))
           #original_img = image_util.load_image_url(img_urls[i], required_size=None, image_type='pil')
            original_img = image_util.load_image_file(img_urls[i], required_size=None, image_type='pil')
        results.append((original_img, f'similarity: {int(dist*100)}%'))

        return results

    