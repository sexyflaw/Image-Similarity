import gradio as gr
import os
import random
from gradio.components import gallery
from gradio.outputs import JSON
from src.model import simlarity_model as model
from src.similarity.similarity import Similarity
from src.util.video import capture_video_frames
import logging
import cv2

logging.basicConfig(level=logging.INFO)
similarity = Similarity()
models = similarity.get_models()
logging.info('结束')

def check(img_main, dir_path, model_idx):
    result = similarity.check_similarity(img_main, dir_path, models[model_idx])
    return result

def check_pic_dir(jie_pic_json, dian_pic_json):
    result = similarity.check_json_similarity(jie_pic_json, dian_pic_json)
    return result

def pic_compute(dir_path, model_idx):
    features = similarity.check_similarity_compute(dir_path, models[model_idx])
    #similarity.pick_diff_img(features)
    return f"计算图片成功{dir_path}"

def video_check(video_path):
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建背景减除器
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # 初始化变量
    frame_count = 0
    last_scene_change = 0
    last_thresh = None
    threshold = 100  # 设置阈值

    # 循环读取视频帧
    while True:
        # 读取下一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 背景减除
        fgmask = fgbg.apply(frame)

        # 二值化处理
        thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]

        # 计算帧差
        if last_thresh is not None:
            # 调整上一帧和当前帧的大小
            last_thresh = cv2.resize(last_thresh, (thresh.shape[1], thresh.shape[0]))
            diff = cv2.absdiff(thresh, last_thresh)

            # 计算场景变化
            if diff.mean() > threshold and frame_count > last_scene_change + 10:
                scene_change = frame_count
                time = scene_change / fps
                print('Scene change detected at time', round(time, 2), 'seconds')
                last_scene_change = scene_change

        # 更新变量
        last_thresh = thresh
        frame_count += 1

    # 释放资源
    cap.release()

# 定义 capture 函数
def capture(video_path):
    output_dir = os.path.dirname(video_path) + '\\captured-%s' % os.path.splitext(os.path.basename(video_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        capture_video_frames(video_path, output_dir)

    return "视频帧捕获完成"

with gr.Blocks() as demo:
    gr.Markdown('视频截图')
    video_path = gr.inputs.Textbox(label='video_path', type='text')
    submit_btn_video = gr.Button('生成视频截图')
    submit_btn_video.click(fn=capture, inputs=[video_path], outputs=gr.outputs.Textbox())

    gr.Markdown('计算截图')
    pic_path = gr.inputs.Textbox(label='pic_path', type='text')
    gr.Markdown('Choose the model')
    model = gr.Dropdown([m.name for m in models], label='Model', type='index')
    submit_btn_pic = gr.Button('计算截图')
    submit_btn_pic.click(fn=pic_compute, inputs=[pic_path, model], outputs=gr.outputs.Textbox())

    gr.Markdown('Checking Image Similarity')
    # img_main = gr.Text(label='Main Image', placeholder='https://myimage.jpg')
    img_main = gr.inputs.Image(label='Main Image', type='filepath')

    gr.Markdown('Images to check')
    # img_1 = gr.Text(label='1st Image', placeholder='https://myimage_1.jpg')
    # img_2 = gr.Text(label='2nd Image', placeholder='https://myimage_2.jpg')
    # img_1 = gr.inputs.Image(label='1st Image', type='filepath')
    # img_2 = gr.inputs.Image(label='2nd Image', type='filepath')
    dir_path = gr.inputs.Textbox(label='dir_path', type='text')

    gr.Markdown('Choose the model')
    model = gr.Dropdown([m.name for m in models], label='Model', type='index')


    submit_btn = gr.Button('Check Similarity')
    submit_btn.click(fn=check, inputs=[img_main, dir_path, model], outputs=gr.outputs.Textbox())

    gr.Markdown('解说截图json')
    jie_pic_json = gr.inputs.Textbox(label='jie_pic_json', type='text')
    gr.Markdown('电影截图json')
    dian_pic_json = gr.inputs.Textbox(label='dian_pic_json', type='text')
    submit_btn = gr.Button('对比电影和解说截图json')
    submit_btn.click(fn=check_pic_dir, inputs=[jie_pic_json, dian_pic_json], outputs=gr.outputs.Textbox())

    gr.Markdown('检测视频切换画面')
    video_path = gr.inputs.Textbox(label='video_path', type='text')
    submit_btn_video = gr.Button('检测视频切换画面')
    submit_btn_video.click(fn=video_check, inputs=[video_path], outputs=gr.outputs.Textbox())

demo.launch()

