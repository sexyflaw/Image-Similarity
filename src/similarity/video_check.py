import json
import os
from scenedetect import detect, ContentDetector
import cv2
import time
import subprocess
import ffmpeg

def deal_scenes(short_path, movie_path, model):
    short_path_new = transcode_video(short_path)
    start = time.time()
    scene_list = detect(short_path_new, ContentDetector(threshold=30))
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
    end = time.time()

    elapsed_time = end - start
    print("解说视频分段耗时：{:.2f}秒".format(elapsed_time))
    return scene_times



def transcode_video(input_path):
    start = time.time()
    input_filename, input_ext = os.path.splitext(input_path)
    output_path = input_filename + '_new' + input_ext

    command = [
        'ffmpeg',
        '-i', input_path,
        '-an',
        '-vf', 'scale=224:224',
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-b:v', '2M',
        output_path
    ]

    subprocess.run(command)
    end = time.time()
    elapsed_time = end - start
    print("视频缩放耗时：{:.2f}秒".format(elapsed_time))
    return output_path

def movie_compute(movie_path, model):
    movie_path_new = transcode_video(movie_path)
    start = time.time()
    cap = cv2.VideoCapture(movie_path_new)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    features_list = []
    frames = []
    frame_indices = []

    progress_interval = frame_count // 10  # 每处理10%的帧打印一次进度

    while True:
        # 读取当前帧
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            break

        # 仅处理每4帧
        if frame_count % 4 == 0:
            frames.append(frame)
            frame_indices.append(frame_count)

        frame_count += 1

        # 打印进度
        if frame_count % progress_interval == 0:
            print("处理进度：{}%".format(frame_count * 100 // progress_interval))

    cap.release()
    print('开始批处理桢')
    # 批量处理帧
    features = model.model_cls.process_frames(frames, frame_indices)

    # 将所有特征添加到features_list
    features_list.extend(features)
    end = time.time()
    elapsed_time = end - start
    print("电影视频计算耗时：{:.2f}秒".format(elapsed_time))

    return features_list


def compute_features(video_file, model):
    frame_step = 10
    reader = ffmpeg.input(video_file)
    reader = reader.filter('select', 'gte(n,{})'.format(frame_step))  # 隔frame_step帧取一帧
    reader = reader.output('pipe:', format='image2', vframes=10)
    process = ffmpeg.run_async(reader, pipe_stdout=True)

    features = []
    frame_indices = []
    frame_count = 0

    while True:
        frame = process.stdout.read()
        if not frame:
            break

        # 处理单帧
        features.append(model.process_frame(frame))
        frame_indices.append(frame_count)

        frame_count += frame_step

    process.wait()

    print("处理完毕,共处理{}帧".format(len(features)))

    return features, frame_indices

# ffmpeg -i D:\download\Video\trans.mp4 -an -vf "scale=224:224" -c:v h264_nvenc -preset fast -b:v 2M D:\download\Video\transnew.mp4