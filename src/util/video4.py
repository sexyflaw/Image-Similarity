import cv2
import json
from multiprocessing import Pool, cpu_count
from collections import defaultdict


def detect_motion(video_a_path, video_b_path, diff_threshold=30, area_threshold=1000, display_frames=True,
                  output_file=None):
    # 加载视频
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)

    # 检查视频尺寸是否一致，如果不一致就调整大小
    width_a, height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_b, height_b = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width_a != width_b or height_a != height_b:
        print("Resizing videos to match their resolutions...")
        if width_a * height_a > width_b * height_b:
            width_a, height_a = width_b, height_b
        else:
            width_b, height_b = width_a, height_a
        resize_a = (width_a, height_a)
        resize_b = (width_b, height_b)
    else:
        resize_a = None
        resize_b = None

    # 读取第一帧
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()

    # 初始化字典，用于保存运动目标的时间段
    motion_objects = defaultdict(list)

    # 循环处理每一帧
    processed_frames = 0
    total_frames = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_interval = total_frames // 10  # 打印进度的间隔帧数
    while ret_a and ret_b:
        processed_frames += 1
        # 打印进度
        if processed_frames % progress_interval == 0:
            progress = processed_frames / total_frames * 100
            print(f"Processed {processed_frames}/{total_frames} frames ({progress:.2f}%)")
        # 调整帧的大小
        if resize_a is not None:
            frame_a = cv2.resize(frame_a, resize_a)
        if resize_b is not None:
            frame_b = cv2.resize(frame_b, resize_b)

        # 对帧进行预处理
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        blur_a = cv2.GaussianBlur(gray_a, (5, 5), 0)
        blur_b = cv2.GaussianBlur(gray_b, (5, 5), 0)

        # 计算帧间差分
        diff = cv2.absdiff(blur_a, blur_b)
        diff[diff < diff_threshold] = 0
        diff[diff >= diff_threshold] = 255

        # 进行形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        # 找出运动目标
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                # 在视频中绘制运动目标
                cv2.rectangle(frame_a, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame_b, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # 记录运动目标在B视频中的时间段
                start_time = cap_b.get(cv2.CAP_PROP_POS_MSEC)
                end_time = start_time + cap_b.get(cv2.CAP_PROP_FRAME_COUNT) / cap_b.get(cv2.CAP_PROP_FPS)
                motion_objects[f"{start_time / 1000:.2f}s-{end_time / 1000:.2f}s"].append(
                    f"{cap_a.get(cv2.CAP_PROP_POS_MSEC) / 1000:.2f}s")

        # 显示帧
        if display_frames:
            cv2.imshow('frame_a', frame_a)
            cv2.imshow('frame_b', frame_b)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 读取下一帧
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

    # 释放资源
    cap_a.release()
    cap_b.release()

    if display_frames:
        cv2.destroyAllWindows()

    # 将运动目标的时间段保存到 JSON 文件中
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(motion_objects, f)

def main():
    video_a_path = 'D:\\download\\Video\\1.mp4'
    video_b_path = 'D:\\download\\Video\\trans.mp4'
    diff_threshold = 30
    area_threshold = 1000
    output_file = 'result.json'

    detect_motion(video_a_path, video_b_path, diff_threshold, area_threshold, False, output_file)

if __name__ == '__main__':
    main()