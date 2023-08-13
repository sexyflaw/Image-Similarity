import os
from PIL import Image
from tqdm import tqdm
import logging
import imagehash

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_phash_similarity(image1, image2):
    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    hamming_distance = hash1 - hash2
    hash_length = hash1.hash.size
    similarity = 1 - (hamming_distance / hash_length)

    return similarity

def find_most_similar_image(target_image_path, directory_path):
    target_image = Image.open(target_image_path)

    most_similar_image_path = None
    max_similarity = -1

    total_files = len(os.listdir(directory_path))
    progress_bar = tqdm(total=total_files, desc="Processing Images")

    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            image = Image.open(image_path)

            similarity = calculate_phash_similarity(target_image, image)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image_path = image_path

            progress_bar.update(1)

    progress_bar.close()

    return most_similar_image_path, max_similarity

logging.getLogger().setLevel(logging.INFO)

target_image_path = 'D:\\download\\Video\\tao_new_frame\\1021.jpg'
directory_path = 'D:\\download\\Video\\trans_new_frame'

most_similar_image, similarity_score = find_most_similar_image(target_image_path, directory_path)

logging.info("最相似的图片路径: %s", most_similar_image)
logging.info("相似度分数: %s", similarity_score)