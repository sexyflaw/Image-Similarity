import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def calculate_similarity(original_image_path, new_image_path):
    # 读取原始图像和待识别的图像
    original_image = cv2.imread(original_image_path)
    new_image = cv2.imread(new_image_path)

    # 调整图像尺寸为模型输入大小
    model_input_size = (224, 224)
    original_image = cv2.resize(original_image, model_input_size)
    new_image = cv2.resize(new_image, model_input_size)

    # 对图像进行预处理
    original_image = preprocess_input(original_image)
    new_image = preprocess_input(new_image)

    # 加载预训练的MobileNet模型
    model = MobileNetV2(weights='imagenet', include_top=False)

    # 提取原始图像和待识别图像的特征向量
    original_features = model.predict(np.expand_dims(original_image, axis=0))
    new_features = model.predict(np.expand_dims(new_image, axis=0))

    # 计算特征向量之间的余弦相似度
    original_features = original_features.reshape(-1)
    new_features = new_features.reshape(-1)
    similarity = np.dot(original_features, new_features) / (np.linalg.norm(original_features) * np.linalg.norm(new_features))

    return similarity

# 调用方法并输出相似度
original_image_path = "D:\\download\\Video\\tao\\imgs\\247.jpg"
new_image_path = "D:\\download\\Video\\tao\\imgs\\290.jpg"
similarity = calculate_similarity(original_image_path, new_image_path)
print("相似度:", similarity)