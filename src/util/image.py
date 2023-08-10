from PIL import Image 
import numpy as np
import requests

def load_image_url(url, required_size = (224,224), image_type = 'array'):
    print(f'downloading.. {url}, type: {image_type}')
    img = Image.open(requests.get(url, stream=True).raw)
    img = Image.fromarray(np.array(img))
    if required_size is not None:
        img = img.resize(required_size)
    if image_type == 'array':
        img = (np.expand_dims(np.array(img), 0)/255).astype(np.float32)
    return img

def load_image_file(file_path,  image_type='array'):
    # print(f'loading.. {file_path}, type: {image_type}')
    img = Image.open(file_path)
    img = Image.fromarray(np.array(img))
    #if required_size is not None:
    #    img = img.resize(required_size)
    if image_type == 'array':
        img = (np.expand_dims(np.array(img), 0)/255).astype(np.float32)
    return img

def load_image_file(file_path,  required_size = (224,224), image_type='array'):
    # print(f'loading.. {file_path}, type: {image_type}')
    img = Image.open(file_path)
    img = Image.fromarray(np.array(img))
    if required_size is not None:
        img = img.resize(required_size)
    if image_type == 'array':
        img = (np.expand_dims(np.array(img), 0)/255).astype(np.float32)
    return img

def load_frame(frame,  required_size = (224,224), image_type='array'):
    img = Image.fromarray(np.array(frame))
    # img = img.resize(required_size)
    if image_type == 'array':
        img = (np.expand_dims(np.array(img), 0)/255).astype(np.float32)
    return img