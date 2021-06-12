import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
import Algo1
import os
from scipy import fftpack
import imageio
from PIL import Image, ImageDraw


def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder, topdown=True):
        for name in sorted(files):
            path = os.path.join(root, name)
            img = cv2.imread(path)
            images.append(img)
    return images



def morph(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening


def low_pass(image):

    # convert image to numpy array
    image1_np = np.array(image)

    # fft of image
    fft1 = fftpack.fftshift(fftpack.fft2(image1_np))

    # Create a low pass filter image
    x, y = image1_np.shape[0], image1_np.shape[1]
    # size of circle
    e_x, e_y = 50, 50
    # create a box
    bbox = ((x / 2) - (e_x / 2), (y / 2) - (e_y / 2), (x / 2) + (e_x / 2), (y / 2) + (e_y / 2))

    low_pass = Image.new("L", (image1_np.shape[0], image1_np.shape[1]), color=0)

    draw1 = ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)

    low_pass_np = np.array(low_pass)

    # multiply both the images
    filtered = np.multiply(fft1, low_pass_np)

    # inverse fft
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))


    return ifft2



def frame_diff(old_frame, new_frame):


    new_frame = cv2.resize(new_frame, (288,288))
    nextframe = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    nextframe = low_pass(nextframe)


    diff = cv2.absdiff(old_frame, nextframe)

    diff = morph(diff,5)
    norm_flow = np.array(diff)
    where_0 = np.where(norm_flow < 8)
    where_1 = np.where(norm_flow > 8)

    norm_flow[where_0] = 0
    norm_flow[where_1] = 1


    #Bounding_box = Algo1.Algorithm1(norm_flow[...,0].transpose(),10)

    return norm_flow
