import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import feature
from skimage import filters
from math import sqrt
from skimage import segmentation
import os, glob

# Declares a set of functions used in the files:
#   1. Fuzzy Detection.ipynb"""


def comparer_duo(img1, img2):
    """
    Plots the two images input into 

    Paramters
    ---------
    img1, img2 : numpy.ndarray
        The image in RGB format. By default the final dimension denotes
        channels
    """
    fig, axs = plt.subplots(1,2, figsize=(8,14), dpi=200)
    imgs = [img1, img2]
    for n, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.imshow(imgs[n])
    fig.tight_layout()

# Iterators to check what works better

def entropy_checker(image):
    '''
    Iterate through footprint values to compare what entropy works 
    better.

    Parameters
    ----------
    image : numpy.ndarray
        The image in RGB format. By default the final dimension denotes
          channels.
    '''
    image_gray = rgb2gray(image)
    f_size = 20
    radi = list(range(1,10))
    fig, ax = plt.subplots(3,3,figsize=(15,15))
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'Radius at {radi[n]}', fontsize = f_size)
        ax.imshow(entropy(img_as_ubyte(image_gray), disk(radi[n])), cmap = 
                  'magma');
        ax.set_axis_off()
    fig.tight_layout()


def threshold_checker(image, entropy_val):
    '''
    Iterate through threshold values to compare what threshold works
    better.

    Parameters
    ----------
    image : numpy.ndarray
        The image in RGB format. By default the final dimension denotes
        channels

    entropy_val 
        The value for the disk to compute image entropy
    '''
    thresholds =  np.arange(0.1,1.1,0.1)
    image_gray = rgb2gray(image)
    entropy_image = entropy(image_gray, disk(entropy_val))
    scaled_entropy = entropy_image / entropy_image.max()  
    fig, ax = plt.subplots(2, 5, figsize=(17, 10))
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'Threshold  : {round(thresholds[n],2)}', 
                     fontsize = 16)
        threshold = scaled_entropy > thresholds[n]
        ax.imshow(threshold, cmap = 'gist_stern_r') ;
        ax.axis('off')
    fig.tight_layout()

# Individual functions related to Image Processing

def entropy_based_thresholder(img, entropy_val, threshold = 0.5):
    '''
    Returns a thresholded masked image for a given entropy disk. The entropy 
    image is scaled between 0 and 1 and compared to the threshold.

    Parameters
    ----------

    img : numpy.ndarray
        The image in RGB format. By default the final dimension denotes
        channels.
    
    entropy val
        The footprint value to compute image entropy.
    
    threshold : float
        Threshold to create a mask for the pixels based on entropy.
    '''
    img_gray = rgb2gray(img)
    entropy_image = entropy(img_gray, disk(entropy_val))
    scaled_entropy = entropy_image / entropy_image.max() # scales the entropy
    mask = scaled_entropy > threshold
    thresholded_img = img_gray * mask
    
    return thresholded_img 

# Utility functions

def open_image_set(path, filetype='png', grayscale = False, selection='all'):
    '''
    Opens a set of images defined at the filepath. Only selects images with 
    the given filetype. 

    Parameters
    ----------

    path : str
        Image filepath.
    
    filetype : 'png', 'jpg', 'tif'
        The format of image files to look for in the path

    grayscale : bool
        Whether returned images should be reduced to grayscale. Default
        False
    
    selection
    '''

    filetypes = ['png', 'jpg', 'tif']
    if filetype not in filetypes:
        raise ValueError("Invalid filetype. Expected one of %s" % filetypes)
    
    imagePaths = [f for f in glob.glob(path + '/*.png')]    # or .jpg, .tif, etc.
    image_set = []

    for n, imagePath in enumerate(imagePaths):
        
        if grayscale == True:
            img = rgb2gray(imread(imagePath))
        else:
            img = imread(imagePath)
            
        image_set.append(img)


    open_images = np.stack(image_set)
    return open_images