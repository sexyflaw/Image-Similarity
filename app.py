import gradio as gr
import os
import random
from src.model import simlarity_model as model
from src.similarity.similarity import Similarity

similarity = Similarity()
models = similarity.get_models()


def check(img_main, img_1, img_2, model_idx):
    result = similarity.check_similarity([img_main, img_1, img_2], models[model_idx])
    return result


with gr.Blocks() as demo:
    gr.Markdown('Checking Image Similarity')
    # img_main = gr.Text(label='Main Image', placeholder='https://myimage.jpg')
    img_main = gr.inputs.Image(label='Main Image', type='filepath')

    gr.Markdown('Images to check')
    # img_1 = gr.Text(label='1st Image', placeholder='https://myimage_1.jpg')
    # img_2 = gr.Text(label='2nd Image', placeholder='https://myimage_2.jpg')
    img_1 = gr.inputs.Image(label='1st Image', type='filepath')
    img_2 = gr.inputs.Image(label='2nd Image', type='filepath')

    gr.Markdown('Choose the model')
    model = gr.Dropdown([m.name for m in models], label='Model', type='index')

    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    ).style(grid=[2], height="auto")

    submit_btn = gr.Button('Check Similarity')
    submit_btn.click(fn=check, inputs=[img_main, img_1, img_2, model], outputs=gallery)

demo.launch()

# 4.6.3 type包
