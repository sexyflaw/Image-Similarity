from typing import List
from scenedetect import detect, ContentDetector

def detect_scenes(video_path: str) -> List[List[str]]:
    scene_list = detect(video_path, ContentDetector())
    scene_times = []

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        scene_times.append([start_time, end_time])

    return scene_times

scene_times = detect_scenes('D:\\download\\Video\\1.mp4')
print(scene_times)