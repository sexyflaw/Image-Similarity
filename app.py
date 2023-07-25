import gradio as gr
import os
import random
from src.model import simlarity_model as model
from src.similarity.similarity import Similarity
import logging

logging.basicConfig(level=logging.INFO)
similarity = Similarity()
models = similarity.get_models()
logging.info('结束')

def check(img_main, dir_path, model_idx):
    result = similarity.check_similarity(img_main, dir_path, models[model_idx])
    return result


with gr.Blocks() as demo:
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

    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    ).style(grid=[1], height="auto")

    submit_btn = gr.Button('Check Similarity')
    submit_btn.click(fn=check, inputs=[img_main, dir_path, model], outputs=gallery)

demo.launch()

# 4.6.3 type包
