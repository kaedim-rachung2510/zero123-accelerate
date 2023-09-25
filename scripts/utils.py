import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

class ImageUtils():

    access_id = "AKIAYFBUHX5MV634PO77"
    access_key = "kUPOe9Xjc441j/0aneRYSu2I/VbLuDlEn+T5Ohyh"

    @staticmethod
    def plot_subplots(images, titles=[], n_cols=4, fig_size=0):
        """Plots the given list of images"""
        n_rows = int(np.ceil(len(images) / n_cols))
        if fig_size == 0:
            fig_size = (2*n_cols, 2.5*n_rows)
        else:
            try:
                len(fig_size)
            except:
                fig_size = (fig_size, fig_size//3*n_rows)
        plt.figure(figsize=fig_size)
        for i,img in enumerate(images):
            plt.subplot(n_rows, n_cols, i+1, xticks=[], yticks=[])
            if len(titles) and titles[i]:
                title = titles[i]
                _title = title if type(title) == str else title[0]
                _color = "k" if type(title) == str else title[1]
                plt.title(_title, color=_color)
            else:
                plt.title(i)
            if type(img) == np.ndarray:
                if np.max(np.array(img)) != 255:
                    plt.imshow((img * 255).astype(np.uint8))
                else:
                    plt.imshow(img.astype(np.uint8))
            else:
                plt.imshow(img)
        try:
            plt.tight_layout()
        except Exception as e:
            print(str(e))
        plt.show()

    # ======= Image Processing =======
    
    @staticmethod
    def convert_to_rgb(img, bg="white"):
        """Pastes rgba image on (white) background"""
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode != "RGBA":
            return img
        return Image.alpha_composite(Image.new(mode="RGBA", size=img.size, color=bg), img).convert("RGB")
    
    @staticmethod
    def resize_keeping_aspect_ratio(img, max_size):
        """Resizes the input image so its length is the given size.
        Doesn't just downsample (ie. is not the same as Image.thumbnail).
        """    
        img_width, img_height = img.size
        if img_width > img_height:
            new_height = int(max_size / img_width * img_height)
            img = img.resize((max_size, new_height))
        else:
            new_width = int(max_size / img_height * img_width)
            img = img.resize((new_width, max_size))
        return img
    
    @staticmethod
    def find_foreground_bounding_box(img, threshold=250):
        """Give the x1,y1,x2,y2 (left-to-right, top-to-bottom) coordinates of the bounding box of the foreground.
        If image is RGB, threshold is the pixel value that indicates something as the background.
        Default background is thresholded at [250,250,250].
        """
        img = np.array(img)

        # mask (image has only 1 channel)
        if img.ndim < 3:
            y, x = img.nonzero() # get the nonzero alpha coordinates
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            return x_min, y_min, x_max, y_max
        
        # RGBA
        if img.shape[2] > 3:
            y, x = img[:,:,3].nonzero() # get the nonzero alpha coordinates
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            return x_min, y_min, x_max, y_max

        min_px_val = threshold # [250,250,250] is considered a background pixel
        h, w = img.shape[:2]
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for col in range(w):
            if not np.all((img[:,col]//min_px_val).ravel()):
                x_min = col
                break
        for col in range(w-1, x_min, -1):
            if not np.all((img[:,col]//min_px_val).ravel()):
                x_max = col
                break
        for row in range(h):
            if not np.all((img[row,:]//min_px_val).ravel()):
                y_min = row
                break
        for row in range(h-1, y_min, -1):
            if not np.all((img[row,:]//min_px_val).ravel()):
                y_max = row
                break
        return x_min, y_min, x_max, y_max
    
    @staticmethod
    def crop_to_foreground(img):
        """Crop the image to just the foreground
        Image mode (RGB/RGBA) is retained.
        """
        if img.mode != "RGBA":
            from rembg import remove
            img = remove(img)
        bounding_box = ImageUtils.find_foreground_bounding_box(img)
        img = img.crop(bounding_box)
        return img
    
    @staticmethod
    def crop_to_square(img, size):
        """Crop the input image to a square image of given length.
        Foreground is identified and resized to 90% of the new size, hence
        keeping aspect ratio but also adding a margin for better performance in segmentation models.
        Image mode (RGB/RGBA) is retained.
        """
        foreground = ImageUtils.crop_to_foreground(img)
        new_foreground = ImageUtils.resize_keeping_aspect_ratio(foreground, int(0.9 * size))
        new_square_img = ImageUtils.paste_centered_on_new_image(new_foreground, (size,size))
        return new_square_img
    
    @staticmethod
    def crop_to_square_by_ratio(img, size, ratio=1):
        """Crop the input image to a square image of given length.
        Foreground is identified and resized to the specified ratio of its original size,
        then pasted on a new image of the given size.
        Image mode (RGB/RGBA) is retained.
        """
        foreground = ImageUtils.crop_to_foreground(img)
        foreground_max_size = np.max(foreground.size)
        foreground = ImageUtils.resize_keeping_aspect_ratio(foreground, int(ratio * foreground_max_size))
        new_square_img = ImageUtils.paste_centered_on_new_image(foreground, (size,size))
        return new_square_img
    
    @staticmethod
    def paste_centered_on_new_image(img_to_paste, img_size, bg="white"):
        """Paste the image on a new white image of given size
        Image mode (RGB/RGBA) is retained.
        """
        img_width, img_height = img_size
        paste_width, paste_height = img_to_paste.size
        new_x_pos = (img_width - paste_width) // 2
        new_y_pos = (img_height - paste_height) // 2
        if img_to_paste.mode == "RGBA":
            bg = (255,255,255,0) if bg == "white" else (0,0,0,0)
        new_img = Image.new(mode=img_to_paste.mode, size=(img_width, img_height), color=bg)
        new_img.paste(img_to_paste, (new_x_pos, new_y_pos))
        return new_img
    
    @staticmethod
    def center_foreground(img):
        """Shifts the foreground so that it's centered.
        Useful when you want the image to retain its dimensions but just shift the foreground to center it.
        Image mode (RGB/RGBA) is retained.
        """
        foreground = ImageUtils.crop_to_foreground(img)
        background = ImageUtils.paste_centered_on_new_image(foreground, img.size)
        return background

    # ======= Image Downloading =======
    
    @staticmethod
    def get_image_from_url(url):
        """Return PIL Image from given url"""
        import requests
        from io import BytesIO
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'}
        response = requests.get(url, headers=headers)       
        img = Image.open(BytesIO(response.content))
        return img

    @staticmethod
    def download_s3_bucket(bucket, path, to):
        import boto3
        s3 = boto3.resource('s3', aws_access_key_id=ImageUtils.access_id,
                            aws_secret_access_key=ImageUtils.access_key, region_name="eu-west-2")
        bucket = s3.Bucket(bucket)
        path = path if path[-1] == "/" else path + "/"
        to = to if to[-1] == "/" else to + "/"
        os.makedirs(to, exist_ok=True)
        count = 0
        for file in bucket.objects.filter(Prefix=path):
            if file.key[-1] == "/":
                continue
            download_location = file.key.replace(path, to)
            if len(download_location.split("/")) > len(to.split("/")):
                folder = "/".join(download_location.split("/")[:-1])
                os.makedirs(folder, exist_ok=True)
            bucket.download_file(file.key, download_location)
            count += 1
        return count
    
    @staticmethod
    def download_image_from_s3(bucket, key, path):
        """Download to path and return PIL image"""
        import boto3
        s3 = boto3.resource('s3', aws_access_key_id=ImageUtils.access_id,
                            aws_secret_access_key=ImageUtils.access_key, region_name="eu-west-2")
        bucket = s3.Bucket(bucket)
        bucket.download_file(key, path)
        return Image.open(path)

class FileUtils():

    @staticmethod
    def list_dir_recursive(dir):
        return [os.path.join(dp, f) for dp, _, fn in os.walk(dir) for f in fn]

    @staticmethod
    def list_files_only(dir):
        return [os.path.join(dir,f) for f in os.listdir(dir) if not os.path.isdir(os.path.join(dir,f))]