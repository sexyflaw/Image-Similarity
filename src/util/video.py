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

def diff_video():


    # 读取电影解说视频截取的片段和电影视频
    cap1 = cv2.VideoCapture('D:\\download\\Video\\1.mp4')
    cap2 = cv2.VideoCapture('D:\\download\\Video\\tao.mp4')

    # 创建背景减除器
    fgbg1 = cv2.createBackgroundSubtractorMOG2()
    fgbg2 = cv2.createBackgroundSubtractorMOG2()

    # 初始化变量
    frame_count1 = 0
    frame_count2 = 0
    last_scene_change1 = 0
    last_scene_change2 = 0
    last_thresh1 = None
    last_thresh2 = None
    threshold = 100  # 设置阈值

    # 循环读取两个视频帧
    while True:
        # 读取下一帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 判断是否读取到帧
        if not ret1 or not ret2:
            break

        # 背景减除
        fgmask1 = fgbg1.apply(frame1)
        fgmask2 = fgbg2.apply(frame2)

        # 二值化处理
        thresh1 = cv2.threshold(fgmask1, 127, 255, cv2.THRESH_BINARY)[1]
        thresh2 = cv2.threshold(fgmask2, 127, 255, cv2.THRESH_BINARY)[1]

        # 计算帧差
        if last_thresh1 is not None and last_thresh2 is not None:
            # 调整上一帧和当前帧的大小
            last_thresh1 = cv2.resize(last_thresh1, (thresh1.shape[1], thresh1.shape[0]))
            last_thresh2 = cv2.resize(last_thresh2, (thresh2.shape[1], thresh2.shape[0]))
            diff1 = cv2.absdiff(thresh1, last_thresh1)
            diff2 = cv2.absdiff(thresh2, last_thresh2)

            # 计算场景变化
            if diff1.mean() > threshold and frame_count1 > last_scene_change1 + 10:
                scene_change1 = frame_count1
                last_scene_change1 = scene_change1
            if diff2.mean() > threshold and frame_count2 > last_scene_change2 + 10:
                scene_change2 = frame_count2
                last_scene_change2 = scene_change2

            # 找到相似片段
            if last_scene_change1 > 0 and last_scene_change2 > 0:
                if last_scene_change1 == last_scene_change2:
                    print('Similar scene found at time', round(last_scene_change1 / 30, 2), 'to',
                          round(frame_count1 / 30, 2), 'seconds in movie video')

        # 更新变量
        last_thresh1 = thresh1
        last_thresh2 = thresh2
        frame_count1 += 1
        frame_count2 += 1

    # 释放资源
    cap1.release()
    cap2.release()

if __name__ == '__main__':
    diff_video()
    #video_path = 'D:\\download\\Video\\1.mp4'
    #output_dir = 'D:\\download\\Video\\%s-captured' % os.path.splitext(os.path.basename(video_path))[0]

    # 确保输出目录存在
    #os.makedirs(output_dir, exist_ok=True)

    #capture_video_frames(video_path, output_dir)