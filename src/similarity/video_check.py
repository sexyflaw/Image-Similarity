import json
import os
from scenedetect import detect, ContentDetector
import cv2
import numpy as np

from src.similarity.model_implements import mobilenet_v3
from src.similarity.model_implements.mobilenet_v3 import ModelnetV3
from src.util import image as image_util


def deal_scenes(short_path, movie_path, model):
    scene_list = detect(short_path, ContentDetector(threshold=30))
    scene_times = []

    total_scenes = len(scene_list)
    for i, scene in enumerate(scene_list):
        start_time = str(scene[0])
        end_time = str(scene[1])
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()

        scene_times.append([start_time, end_time, start_frame, end_frame])

    # Create the output filename by appending "_scene.json" to the base filename
    # base_filename = os.path.splitext(os.path.basename(short_path))[0]
    # output_filename = os.path.join(os.getcwd(), f"{base_filename}_scene.json")
    #
    # # Write the scene times to a JSON file
    # with open(output_filename, 'w') as f:
    #     json.dump(scene_times, f)

    return scene_times


def movie_compute(movie_path, model):
    cap = cv2.VideoCapture(movie_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    features_list = []
    frames = []
    frame_indices = []

    while True:
        # 读取当前帧
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            break

        # 仅处理每30帧
        if frame_count % 30 == 0:
            frames.append(frame)
            frame_indices.append(frame_count)

        frame_count += 1

    cap.release()

    # 批量处理帧
    features = model.model_cls.process_frames(frames, frame_indices)

    # 将所有特征添加到features_list
    features_list.extend(features)

    return features_list