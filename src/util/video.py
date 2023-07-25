import cv2
import datetime
import os


def capture_video_frames(video_path, output_dir):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 设置截图时间间隔（每秒截取5张）
    interval = int(fps / 5)

    # 初始化计数器
    count = 0

    # 获取视频文件名
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 获取视频开始时间
    start_time = datetime.datetime(1970, 1, 1) + datetime.timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        # 检查是否读取到帧
        if not ret:
            break

        # 每隔interval帧截取一张图片
        if count % interval == 0:
            # 调整图像大小为224x224
            resized = cv2.resize(frame, (224, 224))

            # 获取当前帧的时间戳（以毫秒为单位）
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            # 计算当前帧的时间差
            time_delta = datetime.timedelta(milliseconds=timestamp)

            # 计算当前帧的时间
            frame_time = start_time + time_delta - datetime.timedelta(seconds=1)

            # 格式化当前帧的时间为指定格式
            filename = frame_time.strftime('%H-%M-%S,%f')[:-3] + '.jpg'

            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 保存截图
            cv2.imwrite(os.path.join(output_dir, filename), resized)

        # 更新计数器
        count += 1

    # 释放视频文件
    cap.release()

if __name__ == '__main__':
    video_path = 'D:\\download\\Video\\1.mp4'
    output_dir = 'D:\\download\\Video\\%s-captured' % os.path.splitext(os.path.basename(video_path))[0]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    capture_video_frames(video_path, output_dir)