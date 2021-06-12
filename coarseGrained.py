import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
import Algo1
import predictCnn
from PIL import Image
import config
from matplotlib import pyplot
import time
from matplotlib.patches import Rectangle
from scipy import ndimage, misc


# Frame difference with math morphology and filtering

def morph(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening


def crop_pic(bound_box, img):
    images = []
    for box in bound_box:
        crop = img[box[0][1]:box[1][1], box[0][0]:box[1][0]] #11:1079, 96:948
        images.append(crop)

    return images


# https://learnopencv.com/optical-flow-in-opencv/

def dense_optical_flow(method, gauss_sigma, kernel_size, ori_frame, old_frame, params=[], filtering = True, morphing = True):
    # Read the video and first frame
    
    # frame_copy = new_frame
    ori_frame_shape = np.shape(ori_frame)
    #print("ori frame shape", ori_frame_shape)
    new_frame = 0
    # gray scaling and low density filtering of next frame
    if filtering:
        new_frame = cv2.cvtColor(np.float32(ori_frame), cv2.COLOR_BGR2GRAY)
        new_frame = gaussian_filter(new_frame, sigma=gauss_sigma)

    # Calculate Optical Flow
    flow = method(old_frame, new_frame, None, *params)
    u_square, v_square = flow[..., 0]**2, flow[..., 1]**2
    I = u_square+v_square
    I_rooted = np.sqrt(np.array(I))

    # Finding max pixel intensity values
    I_max = I_rooted.max()
    I_min = I_rooted.min()

    # Normalizing
    norm_flow = []
    for i in I_rooted:
        norm_flow.append((i/(I_max-I_min))*255)

    norm_flow = np.array(norm_flow)
    norm_flow = cv2.adaptiveThreshold(norm_flow.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0.5)

    where_0 = np.where(norm_flow == 255)
    where_1 = np.where(norm_flow == 0)

    norm_flow[where_0] = 0
    norm_flow[where_1] = 255


    if morphing:
        norm_flow = ndimage.median_filter(norm_flow, size = 5)
        norm_flow = morph(norm_flow, kernel_size)


    return norm_flow, new_frame



# Gives the binary representation of each frame from the video in the call. Takes the optical flow method from CV2,
# the video, some array of params (from web side) and two booleans on for filtering one for morphology.



