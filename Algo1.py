import numpy as np
import time
import cv2 as cv2

def step1(ex_M, pixel_thresh):
    w = ex_M.shape[1]
    h = ex_M.shape[0]

    # Check for moving part horizontal
    step1 = np.zeros(h)
    for y in range(h):
        zero = np.count_nonzero(ex_M[y])
        if zero > 0:
            step1[y] += 1

        #for x in range(w):
        #    if ex_M[y][x] > 0:
        #        step1[y] += 1

    # Make new array for sepperate moving objects
    idx_arr = init_idx_arr(w, h)
    index_list = []
    ex_list = []
    last_obj = 0
    act_obj = 0

    for i in range(len(ex_M)):
        if (step1[i] > 0):
            if (act_obj == 0):
                last_obj = i
                act_obj = 1

            if (i == len(ex_M) - 1):
                if(i - last_obj < pixel_thresh):
                    act_obj = 0
                    continue
                index_list.append(make_index_arr(idx_arr, last_obj, i + 1, 0, w))
                ex_list.append(make_ex_M(ex_M, last_obj, i + 1, 0, w))
                act_obj = 0

        else:
            if (act_obj == 1):
                if(i - last_obj < pixel_thresh):
                    act_obj = 0
                    continue
                index_list.append(make_index_arr(idx_arr, last_obj, i, 0, w))
                ex_list.append(make_ex_M(ex_M, last_obj, i, 0, w))
                act_obj = 0

    return index_list, ex_list


def step2(idx_list, ex_list, pixel_thresh):
    index2_list = []
    ex2_list = []


    step2_list = []
    for ex_M in ex_list:
        
        h = ex_M.shape[0]
        w = ex_M.shape[1]
        step2 = np.zeros(w)
        for i in range(len(ex_M)):
            for j in range(len(ex_M[i])):
                if ex_M[i][j] > 0:
                    step2[j] += 1
                    
        step2_list.append(step2)

    for x in range(len(ex_list)):
        ex_M = ex_list[x]
        idx_arr = idx_list[x]
        step = step2_list[x]
        last_obj = 0
        act_obj = 0
        h = ex_M.shape[0]
        w = ex_M.shape[1]

        for i in range(len(step)):
            if (step[i] > 0):
                if (act_obj == 0):
                    last_obj = i
                    act_obj = 1

                if (i == len(step) - 1):
                    if(i - last_obj < pixel_thresh):
                        act_obj = 0
                        continue
                    index2_list.append(make_index_arr(idx_arr, 0, h, last_obj, i + 1))
                    ex2_list.append(make_ex_M(ex_M, 0, h, last_obj, i + 1))

            else:
                if (act_obj == 1):
                    if(i - last_obj < pixel_thresh):
                        act_obj = 0
                        continue
                    index2_list.append(make_index_arr(idx_arr, 0, h, last_obj, i))
                    ex2_list.append(make_ex_M(ex_M, 0, h, last_obj, i))
                    act_obj = 0

    return index2_list, ex2_list


def step3(idx_list, ex_list, pixel_thresh):
    index2_list = []
    ex2_list = []

    step2_list = []
    for ex_M in ex_list:
        h = ex_M.shape[0]
        w = ex_M.shape[1]
        step2 = np.zeros(h)
        for i in range(h):
            zero = np.count_nonzero(ex_M[i])
            if zero > 0:
                step2[i] += 1
            #for j in range(w):
            #    if ex_M[i][j] > 0:
            #        step2[i] += 1
                    
        step2_list.append(step2)

    for x in range(len(ex_list)):
        ex_M = ex_list[x]
        idx_arr = idx_list[x]
        step = step2_list[x]
        last_obj = 0
        act_obj = 0
        h = ex_M.shape[0]
        w = ex_M.shape[1]

        for i in range(len(step)):
            if (step[i] > 0):
                if (act_obj == 0):
                    last_obj = i
                    act_obj = 1

                if (i == len(step) - 1):
                    if(i - last_obj < pixel_thresh):
                        act_obj = 0
                        continue
                    index2_list.append(make_index_arr(idx_arr, last_obj, i + 1, 0, w))
                    ex2_list.append(make_ex_M(ex_M, last_obj, i + 1, 0, w))

            else:
                if (act_obj == 1):
                    if(i - last_obj < pixel_thresh):
                        act_obj = 0
                        continue
                    # idx_A = make_index_arr(idx_arr, last_obj, i, 0, w)
                    # print(idx_A)
                    index2_list.append(make_index_arr(idx_arr, last_obj, i, 0, w))
                    ex2_list.append(make_ex_M(ex_M, last_obj, i, 0, w))
                    act_obj = 0

    return index2_list, ex2_list

def make_ex_M(ex_M, w_s, w, h_s, h):
    ex_M_new = np.zeros((w-w_s,h-h_s))
    for i in range(w-w_s):
        for j in range(h-h_s):
            ex_M_new[i][j] = ex_M[i+w_s][j+h_s]
    return ex_M_new

def make_index_arr(idx_arr, w_s, w, h_s, h):
    minh = idx_arr[h_s][w_s][1]
    maxh = idx_arr[h-1][w-1][1]
    minw = idx_arr[h_s][w_s][0]
    maxw = idx_arr[h-1][w-1][0]
    xx = np.arange(minw, maxw+1, 1)
    yy = np.arange(minh, maxh+1, 1)
    index = [[(i,j) for j in yy] for i in xx]
    index = np.array(index, 'i,i')
    return index

def init_idx_arr(w,h):
    xx = np.arange(0, w, 1)
    yy = np.arange(0, h, 1)
    index = [[(i,j) for j in yy] for i in xx]
    return index

def Algorithm1(ex_M, pixel_thresh):
    """
    t0 = time.time()
    index_list, ex_list = step1(ex_M, pixel_thresh)
    t1 = time.time()
    print("Algo step1 time:" , (t1-t0), "amount of boxes:", (len(ex_list)))
    t0 = time.time()
    index_list2, ex_list2 = step2(index_list, ex_list, pixel_thresh)
    t1 = time.time()
    print("Algo step2 time:" , (t1-t0), "amount of boxes:", (len(ex_list2)))
    t0 = time.time()
    index_list3, ex_list3 = step3(index_list2, ex_list2, pixel_thresh)
    t1 = time.time()
    print("Algo step3 time:" , (t1-t0), "amount of boxes:", (len(ex_list3)))
    t0 = time.time()
    """

    index_list, ex_list = step1(ex_M, pixel_thresh)
    index_list2, ex_list2 = step2(index_list, ex_list, pixel_thresh)
    index_list3, ex_list3 = step3(index_list2, ex_list2, pixel_thresh)

    b_box = []
    for x in index_list3:
        minx1 = x[0][0][0], x[0][0][1]
        length = len(x) - 1
        length2 = len(x[length]) -1
        maxx1 = x[length][length2][0], x[length][length2][1]
        b_box.append([minx1, maxx1])
    #print(len(b_box))
    
    """
    t1 = time.time()
    print("Algo step4 time:" + str(t1-t0))
    t0 = time.time()
    """
    return b_box

#sqaurification
def sqaure_crop(c_frame):
    c_frame = c_frame.tolist()
    w, h = len(c_frame), len(c_frame[0])
    diff = 0
    black = [0,0,0]
    if w < h:
        diff = h - w
        #frame_coords[1][0] =
        blacks = [black] * h
        for i in range(diff):
            c_frame.append(blacks)
        w, h = len(c_frame), len(c_frame[0])
        bbox = [w,h]
        c_frame = np.array(c_frame).astype(np.uint8)
        return c_frame, bbox

    else:
        diff = w - h
        for i in range(w):
            for _ in range((w-1), (diff + w -1)):
                c_frame[i].append(black)
    c_frame = np.array(c_frame).astype(np.uint8)
    w, h = len(c_frame), len(c_frame[0])
    bbox = [w,h]
    return c_frame, bbox
    """
    else:
        diff = w - h
        blacks = [black] * diff
        for i in range(w):
            c_frame[i].append(blacks)
    bbox = [[0,0],[w,h]]
    c_frame = np.array(c_frame)
    c_frame = c_frame.astype(np.uint8)
    return c_frame, bbox
    """

def size_up(c_frame, size):
    w, h = len(c_frame), len(c_frame[0])

    full = np.full((size, size, 3), (0,0,0),dtype=np.uint8)
    
    xx = (size - w) // 2
    yy = (size - h) // 2

    full[xx:xx+w, yy:yy+h] = c_frame
    bbox = [yy,xx]
    return full, bbox

"""def scale_crop_bbox(bound_box, img, scale):
    bboxes = []
    h, w, c = np.shape(img)
    for box in bound_box:
        height = box[1][1]
        y_off = box[0][1]
        width = box[1][0]
        x_off = box[0][0]
        
        height = box[1][1]
        y_off = box[0][1]
        down_scale = (1-(1 - scale))
        if math.floor(box[0][1] - down_scale * (height - box[0][1])) > 0:
            y_off = math.floor(box[0][1] - (1 - scale) * (height - box[0][1]))
        
        if math.floor(box[1][1] * scale) < h:
            height = math.floor(box[1][1] * scale)
        width = box[1][0]
        x_off = box[0][0] 
        if math.floor(box[0][0] - down_scale * (width - box[0][0])) > 0:
            x_off = math.floor(box[0][0] - (1 - scale) * (width - box[0][0]))
        if math.floor(box[1][0] * scale) < w:
            width = math.floor(box[1][0] * scale)
        bboxes.append([(x_off, width),(y_off, height)])
    return bboxes"""

# https://learnopencv.com/optical-flow-in-opencv/
def scale_bboxes(bbox, ori_frame_shape, frame_box, sqaure_shape):
    #Scales bounding boxes to original frame from 
    #bounding boxes given by model from cropped squared frame
    scaled_bboxes = []
    for box in bbox:
        x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
        ori_y2, ori_x2, channels = ori_frame_shape
        h_offset, w_offset = sqaure_shape 
       
        x1_coord = (x1 * 416) - h_offset
        y1_coord = (y1 * 416) - w_offset
        width = x2 * 416
        height = y2 * 416
        

        x1_n = (x1_coord + frame_box[0][0]) / ori_x2
        y1_n = (y1_coord + frame_box[0][1]) / ori_y2
        x2_n = width / ori_x2
        if box[0] == 0:
            x2_n = x2_n * 0.85
        y2_n = height / ori_y2

        box[2], box[3], box[4], box[5] = x1_n, y1_n, x2_n, y2_n 
        scaled_bboxes.append(box)

    return scaled_bboxes[0]