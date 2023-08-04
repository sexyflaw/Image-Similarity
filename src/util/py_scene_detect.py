import json
from typing import List
from scenedetect import detect, ContentDetector
import os
import cv2

def detect_scenes(video_path: str) -> List[List[str]]:
    scene_list = detect(video_path, ContentDetector())
    scene_times = []

    for i, scene in enumerate(scene_list):
        start_time = str(scene[0])
        end_time = str(scene[1])
        scene_times.append([start_time, end_time])

        # Save the frame as JPG for each scene
        capture = cv2.VideoCapture(video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, scene[0].get_frames())

        # Get the directory and base filename of the input video file
        video_dir, video_filename = os.path.split(video_path)
        video_basename, video_ext = os.path.splitext(video_filename)
        output_dir = os.path.join(video_dir, video_basename + '_scene')

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the frame as JPG
        frame_time = scene[0].get_timecode().replace(':', '_')
        frame_filename = os.path.join(output_dir, f"{frame_time}.jpg")
        success, frame = capture.read()
        if success:
            cv2.imwrite(frame_filename, frame)
            print("Saved frame:", frame_filename)
        else:
            print("Failed to capture frame:", frame_filename)

        capture.release()

    # Create the output filename by appending "_scene.json" to the base filename
    output_filename = os.path.join(video_dir, video_basename + '_scene.json')

    # Write the scene times to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(scene_times, f)

    return scene_times

scene_times = detect_scenes('D:\\download\\Video\\trans.mp4')
print(scene_times)