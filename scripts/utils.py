import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

def plot_subplots(images, titles=[], n_cols=4, fig_size=0):
    if fig_size == 0:
        fig_size = 2 * n_cols
    n_rows = int(np.ceil(len(images) / n_cols))
    plt.figure(figsize=(fig_size, fig_size//3*n_rows))
    for i,img in enumerate(images):
        plt.subplot(n_rows, n_cols, i+1, xticks=[], yticks=[])
        if len(titles) and titles[i]:
            title = titles[i]
            _title = title if type(title) == str else title[0]
            _color = "k" if type(title) == str else title[1]
            plt.title(_title, color=_color)
        else:
            plt.title(i)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()

def convert_to_rgb(img):
    if img.mode != "RGBA":
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, "white"), img).convert("RGB")

def resize_keeping_aspect_ratio(img, size):
    """Resizes the input image to the specified max size while keeping the aspect ratio
    Input should be an ndarray
    """
    img_height, img_width = img.shape[:2]
    if img_width > img_height:
        new_height = int(size / img_width * img_height)
        img = cv2.resize(img, (size,new_height), interpolation=cv2.INTER_AREA)

    else:
        new_width = int(size / img_height * img_width)
        img = cv2.resize(img, (new_width,size), interpolation=cv2.INTER_AREA)

    return img

def resize_square_keeping_aspect_ratio(img, size, bg=1):
    """Resizes the input image to a square while keeping the aspect ratio
    and padding surrounding areas with black
    Input should be an ndarray
    """
    img_height, img_width = img.shape[:2]
    if img_width == img_height:
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
        return img

    elif img_width > img_height:
        new_height = int(size / img_width * img_height)
        img = cv2.resize(img, (size,new_height), interpolation=cv2.INTER_AREA)
        padding = 255 * np.ones(((size-new_height)//2, size)).astype(np.uint8) * bg
        padding = np.tile(padding, (img.shape[2],1,1)).transpose(1,2,0)
        img = np.concatenate((padding, img.astype(np.uint8), padding), axis=0)

    else:
        new_width = int(size / img_height * img_width)
        img = cv2.resize(img, (new_width,size), interpolation=cv2.INTER_AREA)
        padding = 255 * np.ones((size, (size-new_width)//2)).astype(np.uint8) * bg
        padding = np.tile(padding, (img.shape[2],1,1)).transpose(1,2,0)
        img = np.concatenate((padding, img.astype(np.uint8), padding), axis=1)

    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    return img

def crop_to_square(img, size):
    """Crop the input image to a square
    Input should be an ndarray
    """
    img_height, img_width = img.shape[:2]
    if img_width > img_height:
        margin = int((img_width - img_height) / 2)
        img = img[:, margin:margin+img_height]
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)

    elif img_height > img_width:
        margin = int((img_height - img_width) / 2)
        img = img[margin:margin+img_width, :]

    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    return img

def get_image_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'}
    response = requests.get(url, headers=headers)       
    img = Image.open(BytesIO(response.content))
    return img