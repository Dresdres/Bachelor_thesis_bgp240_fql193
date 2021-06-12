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
import os
import utils
from scipy import ndimage, misc

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

def alt_get_bounding_boxes_labels(frame):
    boxes = []
    gray_data = []
    dice_labels = []
    temp = []
    w = frame.shape[1]
    h = frame.shape[0]

    gray_data.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray_data = np.array(gray_data)

    for i in range(len(gray_data)):
        boxes.append(Algo1.Algorithm1(gray_data[i], 10))

    for box in boxes:
        boxes = []
        for b in box:
            boxes.append([b[0][0], b[0][1], b[1][0] , b[1][1]])
        dice_labels.append(boxes)
    for Other_bounding_box in dice_labels:
        Other_bounding_box = []
        for i in range(len(dice_labels)):
            if np.shape(dice_labels[i]) == (0,):
                Other_bounding_box.append([])

            elif ((dice_labels[i][0][2]) - (dice_labels[i][0][0])) * ((dice_labels[i][0][3]) - (dice_labels[i][0][1])) > 5:
                Other_bounding_box.append(dice_labels[i])

    return Other_bounding_box


def alt_get_bounding_pred_labels(frame, ori_frame):
    boxes = []
    gray_data = []
    dice_labels = []
    labels = []
    box1 = []
    gray_data.append(frame)
    gray_data = np.array(gray_data)


    for i in range(len(gray_data)):
        boxes.append(Algo1.Algorithm1(gray_data[i], 10))


    for box in boxes:
        boxes = []
        for b in box:
            boxes.append([b[0][0], b[0][1], b[1][0], b[1][1]])

        np.array(dice_labels.append(boxes))

        for Other_bounding_box in dice_labels:
            Other_bounding_box = []
        for i in range(len(dice_labels)):
            if np.shape(dice_labels[i]) == (0,):
                Other_bounding_box.append([])
            elif (dice_labels[i][0][2] - dice_labels[i][0][0]) * (dice_labels[i][0][3] - dice_labels[i][0][1]) > 2500:

                Other_bounding_box.append(dice_labels[i])


            else:
                Other_bounding_box.append([])


    for i in range(len(Other_bounding_box[0])):
        if len(Other_bounding_box[0])==0:
            continue
        else:
            Other_bounding_box[0][i][0] = Other_bounding_box[0][i][0]*0.80
            Other_bounding_box[0][i][1] = Other_bounding_box[0][i][1] * 0.80
            if Other_bounding_box[0][i][2] * 1.20 < np.shape(frame)[0]:
                Other_bounding_box[0][i][2] = Other_bounding_box[0][i][2] * 1.20
            else:
                Other_bounding_box[0][i][2] = Other_bounding_box[0][i][2]
            if Other_bounding_box[0][i][3] *1.20 <np.shape(frame)[1]:
                Other_bounding_box[0][i][3] = Other_bounding_box[0][i][3] * 1.20
            else:
                Other_bounding_box[0][i][3] = Other_bounding_box[0][i][3]

    print(len(Other_bounding_box[0]))
    print(Other_bounding_box)
    if Other_bounding_box != [[]]:
        for i in range(len(Other_bounding_box[0])):
            color = (255,0,0)
            thickness = 2

            x = (math.ceil(Other_bounding_box[0][i][0]), math.ceil(Other_bounding_box[0][i][1]))
            y = (math.ceil(Other_bounding_box[0][i][2]), math.ceil(Other_bounding_box[0][i][3]))
            image = cv2.rectangle(ori_frame, x, y, color, thickness)
            cv2.imshow("test", image)
            k = cv2.waitKey(50000) & 0xFF
            if k == 27:
                exit

    else:
        return Other_bounding_box


def dense_optical_flow(method, gauss_sigma, kernel_size, video_path,  label_path, params=[], filtering = True, morphing = True):
    # Read the video and first frame
    data_set = load_images_from_folder((video_path))
    label_set = load_images_from_folder((label_path))
    label_set_np = np.array(label_set)
    data_set = np.array(data_set)

    all_frame_box = []
    final_array = []
    model = 0
    count = 0
    old_frame = data_set[count]


    #gray scaling and low density filtering of next frame
    if filtering:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # old_frame = low_pass_filter(old_frame)
        old_frame = gaussian_filter(old_frame, sigma=gauss_sigma)

    while count <len(data_set):
        # Read the next frame
        if count == len(data_set)-1:
            new_frame = data_set[count]
        else:
            new_frame = data_set[count+1]
        ori_frame = new_frame
        # frame_copy = new_frame


        # gray scaling and low density filtering of next frame
        if filtering:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            # new_frame = low_pass_filter(new_frame)
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
        norm_flow = cv2.adaptiveThreshold(norm_flow.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 0.5)

        where_0 = np.where(norm_flow == 255)
        where_1 = np.where(norm_flow == 0)

        norm_flow[where_0] = 0
        norm_flow[where_1] = 255


        if morphing:
            norm_flow = ndimage.median_filter(norm_flow, size=10)
            norm_flow = morph(norm_flow, kernel_size)




        final_array.append(norm_flow)
        count += 1


        old_frame = new_frame


    return final_array

def score(true_box, pred_box):
    dice = 0
    count = 0
    if len(true_box[0]) < len(pred_box[0]):
        for i in range(len(pred_box[0])):
            try:
                if (true_box[0] == [] or true_box[0] == [[]]) and (pred_box[0] != [] or pred_box[0] != [[]]):
                    count += 1
                elif (true_box[0] == [] or true_box[0] == [[]]) and (pred_box[0] == [] or pred_box[0] == [[]]):
                    count +=1
                    dice +=1
                else:
                    try:
                        right_x_true, right_y_true, left_x_true, left_y_true = true_box[0][i][0], true_box[0][i][1], true_box[0][i][2], true_box[0][i][3]
                        right_x_pred, right_y_pred, left_x_pred, left_y_pred = pred_box[0][i][0], pred_box[0][i][1], pred_box[0][i][2], pred_box[0][i][3]

                        if right_x_pred < right_x_true and right_y_pred < right_y_true and left_x_pred > left_x_true and left_y_pred > left_y_true:
                            count += 1
                            dice += 1
                        else:
                            count+=1
                    except:
                        count +=1
            except:
                count += 1

    elif len(true_box[0]) > len(pred_box[0]):
        for i in range(len(true_box[0])):
            if (pred_box[0] == [] or pred_box[0] == [[]]) and (true_box[0]!= [] or true_box[0] != [[]]):
                count += 1
            elif ((true_box[0] == [] or true_box[0] == [[]])) and (pred_box[0] == [] or pred_box[0] == [[]]):
                count += 1
                dice += 1
            else:
                try:
                    left_x_true, left_y_true, right_x_true, right_y_true = true_box[0][i][0], true_box[0][i][1], true_box[0][i][2], true_box[0][i][3]
                    left_x_pred, left_y_pred, right_x_pred, right_y_pred = pred_box[0][i][0], pred_box[0][i][1], pred_box[0][i][2], pred_box[0][i][3]

                    if left_x_pred <= left_x_true and left_y_pred <= left_y_true and right_x_pred >= right_x_true and right_y_pred >= right_y_true:
                        count += 1
                        dice += 1
                    else:
                        count+=1
                except:
                    count +=1

    else:
        if len(true_box[0])==0:
            dice +=1
            count +=1
        else:
            for i in range(len(true_box[0])):
                if (true_box[0] == [] or true_box[0] == [[]]) and (pred_box[0] != [] or pred_box[0] != [[]]):
                    count += 1
                elif ((true_box[0] == [] or true_box[0] == [[]])) and (
                        pred_box[0] == [] or pred_box[0] == [[]]):
                    count += 1
                    dice += 1
                elif pred_box[0][i][0] <= true_box[0][i][0] and pred_box[0][i][1] <= true_box[0][i][1] and pred_box[0][i][2] >= true_box[0][i][2] and pred_box[0][i][3] >= true_box[0][i][3]:
                    count +=1
                    dice +=1
                else:
                    count +=1
    return dice/count

def dice_score(true_box, pred_box):
    """
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
    """

    dice = 0
    intersec = 0
    if true_box == [[]] and pred_box != [[]]:
        dice += 0

    elif true_box == [[]] and pred_box == [[]]:
        dice +=1

    elif (true_box == [] or true_box == [[]]) and (pred_box == [] or pred_box == [[]]):
        dice += 1

    elif (pred_box == [] or pred_box == [[]]) and true_box != [[]]:
        dice += 0

    elif pred_box != [] and true_box == []:
        dice += 0

    else:
        dx = min(true_box[0][0][2], pred_box[0][0][2]) - max(true_box[0][0][0], pred_box[0][0][0])
        dy = min(true_box[0][0][3], pred_box[0][0][3]) - max(true_box[0][0][1], pred_box[0][0][1])


        if (dx >= 0) and (dy >= 0):
            intersec = 2 * (dx * dy)
            areas = rect_area(true_box) + rect_area(pred_box)
            dice += intersec / areas

    return dice  # Returns dice score of two bboxes


def rect_area(box):
    x = box[0][0][2] - box[0][0][0]
    y = box[0][0][3] - box[0][0][1]
    return x * y

def coarse_grain_trainer(pred, target):
    error = 0
    for i in range(len(pred)):
        if np.shape(pred[i]) != np.shape(target[i]):
            error +=1

    return 1-(error/len(pred))


def coarse_grain_search(data_set_path, label_path, tuning_param1, tuning_param2, tuning_param3, tuning_param4, tuning_param5, tuning_param6, tuning_param7):

    pred_bboxes = []
    bboxes = []
    error = 0
    target = dense_optical_flow(cv2.calcOpticalFlowFarneback, tuning_param1, tuning_param2, data_set_path, label_path, [0.5, tuning_param5, tuning_param3, tuning_param6, tuning_param7, tuning_param4, 0], True, True)
    label_set = load_images_from_folder(label_path)
    label_set_np = np.array(label_set)
    data_set = load_images_from_folder(data_set_path)
    data_set_np = np.array(data_set)

    for data in label_set_np:
        bboxes.append(alt_get_bounding_boxes_labels(data))
    for i in range(len(target)):
        pred_bboxes.append(alt_get_bounding_pred_labels(target[i], data_set_np[i]))
    for i in range(len(pred_bboxes)):
        if pred_bboxes[i] == [] and (bboxes[i] == [[]] or bboxes[i] == []):
            error += 1
        else:
            error += score(bboxes[i], pred_bboxes[i])
    return error/len(label_set_np)


path = '/home/jonathan/School/Bachelor/clonehub/Bach_bgp240_fql193_V2/unet_stuff/O_CL_02'
label_path = '/home/jonathan/School/Bachelor/clonehub/Bach_bgp240_fql193_V2/unet_stuff/O_CL_02-GT'

print("test of Coarse Grain",coarse_grain_search(path, label_path,0.1,5,20,1.1,10,10,10))

# for grid search
'''sigma = [ 0.1, 1,5]
kernel = [3,5]
pyramid_size = [10,15,20,100]
param = [1.1]
tuning_param5 = [2,15]
tuning_param6 = [2,15]
tuning_param7 = [2,15]

for i in sigma:
    for j in kernel:
        for k in pyramid_size:
            for m in param:
                for n in tuning_param5:
                    for p in tuning_param6:
                        for o in tuning_param7:
                            print("sigma value : " + str(i) + " kernel size : " + str(j) + " param : " + str(m) +" pyramid_size : " + str(k)  +
                                        " tuning_param5 : " + str(n) + " tuning_param6 : " + str(p) +  " tuning_param7 : " + str(o) + " = ",coarse_grain_search(path, label_path,i,j,k,m,n,p,o))

exit()'''
# optimal so far sigma value : 0.05 kernel size : 5 param : 1.1 pyramid_size : 10 tuning_param5 : 3 tuning_param6 : 3 tuning_param7 : 7, with 2000 thresh and without upscaling 69 percent
