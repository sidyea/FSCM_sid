import numpy as np
import skimage
from skimage.util import crop
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as Image
import os
import glob
import scipy.misc as sm


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def load_data(dir_name):
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    import glob
    imagePaths = [f for f in glob.glob(dir_name + '/*.png')]  # or .jpg, .tif, etc
    for filename in imagePaths:
        img = Image.open(filename)
        img = crop(img, ((400, 180), (420, 420), (0, 0)), copy=False)
        img = rgb2gray(img)
        imgs.append(img)
    return imgs


def visualize(imgs, format=None, gray=False):
    
    plt.figure(figsize=(20, 20))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        plt_idx = i + 1
        plt.subplot(4, 4, plt_idx)
        plt.imshow(img, format)
    plt.show()

