import cv2
import os
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def find_most_similar_image(target_image_path, image_directory):
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    max_ssim = -1
    most_similar_image = None

    total_images = len(os.listdir(image_directory))

    with tqdm(total=total_images, desc="Comparing images") as pbar:
        for image_file in os.listdir(image_directory):
            image_path = os.path.join(image_directory, image_file)
            if os.path.isfile(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    ssim_score = ssim(target_image, image)
                    if ssim_score > max_ssim:
                        max_ssim = ssim_score
                        most_similar_image = image_path
                        print(max_ssim,most_similar_image)

            pbar.update(1)

    return most_similar_image



# 示例用法
target_image_path = "D:\\download\\Video\\tao_new_frame\\1021.jpg"
image_directory = "D:\\download\\Video\\trans_new_frame"
most_similar_image = find_most_similar_image(target_image_path, image_directory)
if most_similar_image is not None:
    print("Most similar image:", most_similar_image)
else:
    print("No similar image found.")