import gradio as gr
from src.similarity import video_check
from src.similarity.similarity import Similarity
import logging


logging.basicConfig(level=logging.INFO)
similarity = Similarity()
models = similarity.get_models()
logging.info('结束')


def check(short_video_path, movie_video_path):
    # 解说视频分段
    result = video_check.deal_scenes(short_video_path, movie_video_path, models[0])
    # 电影视频计算特征
    features_list = video_check.movie_compute(movie_video_path, models[0])
    # 解说分段画面计算相似度

    return result


with gr.Blocks() as demo:

    gr.Markdown('解说视频路径')
    short_path = gr.inputs.Textbox(label='short_path', type='text')

    gr.Markdown('电影视频路径')
    movie_path = gr.inputs.Textbox(label='movie_path', type='text')

    submit_btn = gr.Button('运行')
    submit_btn.click(fn=check, inputs=[short_path, movie_path], outputs=gr.outputs.Textbox())
    print('运行结束')

demo.launch()
