import json
from typing import List
from scenedetect import detect, ContentDetector
import os

def detect_scenes(video_path: str) -> List[List[str]]:
    scene_list = detect(video_path, ContentDetector())
    scene_times = []

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        scene_times.append([start_time, end_time])

    # Get the directory and base filename of the input video file
    video_dir, video_filename = os.path.split(video_path)
    video_basename, video_ext = os.path.splitext(video_filename)

    # Create the output filename by appending "_scene.json" to the base filename
    output_filename = os.path.join(video_dir, video_basename + '_scene.json')

    # Write the scene times to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(scene_times, f)

    return scene_times

scene_times = detect_scenes('D:\\download\\Video\\tao.mp4')
print(scene_times)