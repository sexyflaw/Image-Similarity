import gradio as gr
import os
import random
from gradio.components import gallery
from gradio.outputs import JSON
from src.model import simlarity_model as model
from src.similarity.similarity import Similarity
from src.util.video import capture_video_frames
import logging

logging.basicConfig(level=logging.INFO)
similarity = Similarity()
models = similarity.get_models()
logging.info('结束')

def check(img_main, dir_path, model_idx):
    result = similarity.check_similarity(img_main, dir_path, models[model_idx])
    return result

def pic_compute(dir_path, model_idx):
    result = similarity.check_similarity_compute(dir_path, models[model_idx])
    return f"计算图片成功{dir_path}"


# 定义 capture 函数
def capture(video_path):
    output_dir = os.path.dirname(video_path) + '\\captured-%s' % os.path.splitext(os.path.basename(video_path))[0]

    # 如果输出目录不为空，则跳过捕获视频帧的步骤
    if os.listdir(output_dir):
        print('Output directory is not empty. Skipping capture_video_frames().')
    else:
        # 否则，捕获视频帧
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

demo.launch()

