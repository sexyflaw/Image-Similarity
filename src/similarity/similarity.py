from src.model import simlarity_model as model
from src.util import image as image_util
from src.util import matrix
from .model_implements.mobilenet_v3 import ModelnetV3
from .model_implements.vit_base import VitBase
from .model_implements.bit import BigTransfer
import os
import json
import numpy as np
from scipy.spatial import distance_matrix
import logging


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
            # imgs.append(image_util.load_image_file(url, required_size=(model.image_size, model.image_size),
                                                 #  image_type=model.image_input_type))
            imgs.append(image_util.load_image_file(url, image_type=model.image_input_type))
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
        file_name = os.path.splitext(os.path.basename(dir_path))[0]
        output_file = os.path.join(parent_dir, file_name + "_imgDataJson.json") # 拼接文件路径
        if os.path.exists(output_file):
            print(f"文件 {output_file} 存在")
            with open(output_file, 'r') as f:
                features_list = json.load(f)

            # 将列表中的数据转换回字典对象
            features = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in features_list]
        else:
            print(f"文件 {output_file} 不存在")
            #image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                           #os.path.isfile(os.path.join(dir_path, f))]

            # 获取文件夹中的所有文件名，并按照升序排序
            file_names = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

            # 生成文件的绝对路径
            image_files = [os.path.join(dir_path, f) for f in file_names]
            print('计算图片开始')
            features = model.model_cls.extract_feature_dictV2(image_files)
            # 将字典对象中的 ndarray 对象转换为列表
            features_list = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()} for d in features]

            # 写入数据到 JSON 文件
            with open(output_file, 'w') as f:
                json.dump(features_list, f)

            print(f"Feature data saved to file: {output_file}")

        return features

    def pick_diff_img (self, features):
        print('pick_diff_img')

        similar_pairs = []

        for i, fv1 in enumerate(features):
            for j in range(i + 1, len(features)):
                fv2 = features[j]
                dist = matrix.cosine(fv1["feature"], fv2["feature"])
                print(f"The cosine distance between fv{i} and fv{j} is {dist:.4f}")

                if dist < 0.9:
                    # 如果距离小于 0.9，则记录相似的图像文件名
                    similar_pairs.append((fv1["filename"], fv2["filename"]))
                    i = j

        print("Similar pairs:", similar_pairs)

    def check_json_similarity(self,jie_pic_json,dian_pic_json):
        if os.path.exists(jie_pic_json):
            print(f"文件 {jie_pic_json} 存在")
            with open(jie_pic_json, 'r') as f:
                jie_features_list = json.load(f)

            # 将列表中的数据转换回字典对象
            jie_features = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in jie_features_list]
        else:
            raise FileNotFoundError("解说文件不存在")

        if os.path.exists(dian_pic_json):
            print(f"文件 {dian_pic_json} 存在")
            with open(dian_pic_json, 'r') as f:
                dian_features_list = json.load(f)

            # 将列表中的数据转换回字典对象
            dian_features = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in dian_features_list]
        else:
            raise FileNotFoundError("电影文件不存在")

        results = []
        print("开始对比数据")
        for j, y in enumerate(jie_features):
            lastDist = 0
            lastFile = ""
            for i, v in enumerate(dian_features):
                dist = matrix.cosine(y["feature"], v["feature"])
                if dist > lastDist:
                    lastDist = dist
                    lastFile = v["filename"]
                    # print(f'{i} -- distance: {dist}--last-file: {lastFile}')
                # results.append((imgs[i], f'similarity: {int(dist*100)}%'))
                # original_img = image_util.load_image_url(img_urls[i], required_size=None, image_type='pil')
                # original_img = image_util.load_image_file(image_files[i], required_size=None, image_tytpe='pil')
            print(f'{y["filename"]} -- distance: {dist}--last-file: {lastFile}')
            results.append(y["filename"], lastFile, {dist})
        print("end of results")
        output_file = os.path.dirname(jie_pic_json)
        compare_json_file = os.path.join(output_file,"compare.json")
        # 写入数据到 JSON 文件
        with open(compare_json_file, 'w') as f:
            json.dump(results, f)

        print(f"Feature data saved to file: {compare_json_file}")
        return results