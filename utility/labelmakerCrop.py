import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import Algo1
from PIL import Image
from matplotlib import pyplot
import time
from matplotlib.patches import Rectangle
import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split

def get_bounding_boxes_labels(path, object):
    boxes = []
    labels = []
    data_set = load_images_from_folder((path))
    data_set_np = np.array(data_set)
    gray_data = []


    for i in range(len(data_set_np)):
        gray_data.append(cv2.cvtColor(data_set_np[i], cv2.COLOR_BGR2GRAY))
    gray_data = np.array(gray_data)
    for i in range(len(gray_data)):
        boxes.append(Algo1.Algorithm1(gray_data[i],5))
        print(i)
    dice_labels = []
    for box in boxes:
        if len(box) == 0:
            dice_labels.append(" ")
        elif len(box) == 1:
            dice_labels.append([box[0][0][0]/352, box[0][0][1]/288, box[0][1][0]/352, box[0][1][1]/288])
        elif len(box) == 2:
            dice_labels.append([[box[0][0][0]/352 ,box[0][0][1]/288, box[0][1][0]/352, box[0][1][1]/288],
                        [box[1][0][0] / 352, box[1][0][1] / 288 ,box[1][1][0] / 352 ,
                            box[1][1][1] / 288]])
        else:
            dice_labels.append([[[box[0][0][0]/352, box[0][0][1]/288, box[0][1][0]/352, box[0][1][1]/288],
                        [box[1][0][0] / 352, box[1][0][1] / 288, box[1][1][0] / 352,
                            box[1][1][1] / 288],[box[2][0][0] / 352,box[2][0][1] / 288,box[2][1][0] / 352,
                            box[2][1][1] / 288]]])
    # printer eksempel på et frame med to bounding boxes.
    print(dice_labels[251])
    return dice_labels
                

def alt_get_bounding_boxes_labels(frame, cls):
    boxes = []
    gray_data = []
    dice_labels = []
    labels = []

    h = np.shape(frame)[0]
    w = np.shape(frame)[1]
    gray_data.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    gray_data = np.array(gray_data)

    for i in range(len(gray_data)):
        boxes.append(Algo1.Algorithm1(gray_data[i],15))
    for box in boxes:
        boxes = []
        for b in box:
            width =  (b[1][0]/w - b[0][0]/w) 
            height = (b[1][1]/h - b[0][1]/h) 
            b1 =  (b[1][0]/w -  b[0][0]/w) /2 +  b[0][0]/w #(b[0][0]/w + b[1][0]/w) / 2
            b2 = (b[1][1]/h -  b[0][1]/h) /2 +  b[0][1]/h
            boxes.append([cls, b1, b2, width, height])

        dice_labels.append(boxes)
        

    return dice_labels

def load_images_from_folder(label_path):
    names = []
    images = []
    for root, dirs, files in os.walk(label_path, topdown=True):
        for name in files:
            path = os.path.join(root, name)
            img = cv2.imread(path)
            names.append(name)
            images.append(img)
    return images, names

def save_bboxes_txt(labels, names, label_path):
    empty_counter = 0
    for i in range(len(labels)):
        file_name = names[i][0:14] + "-" + [i] + ".txt"
 
        completeName = os.path.join(label_path, file_name)
        label = labels[i][0]
        file = open(completeName,  "w")
        if (len(label) != 0) and (len(label) != 1):
            for l in label:
                file.writelines(("", str(l[0]), " ", str(l[1]), " ", 
                    str(l[2]), " ", str(l[3]), " ", str(l[4]), "\n"))
            file.close()
            continue
        try:
            assertion = np.shape(label)[1]
        except:
            empty_counter += 1
            file.close()
            continue

        if np.shape(label)[1] == 5:
            label = label[0]
            file.writelines(("", str(label[0]), " ", str(label[1]), " ",
                 str(label[2]), " ", str(label[3]), " ", str(label[4])))
            file.close()
            continue
        
        file.close()
    print("Empty counter:", empty_counter)
    return 1
    """
    #uncomment to view example of box.
    print(np.shape(boxes))
    print(boxes[104])
    image = cv2.rectangle(gray_data[104], boxes[104][0][1], boxes[104][0][0], (255, 0, 0), 2)
    for b_box in boxes[104]:
        image = cv2.rectangle(data_set_np[104], b_box[0], b_box[1], (255,0,0),2)
    cv2.imshow("test", image)
    k = cv2.waitKey(1000) & 0xFF
    if k == 1:
        exit()
    return labels
    """

def change_names(label_path):
    for root, dirs, files in os.walk(label_path, topdown=True):
        for name in files:
                path = os.path.join(root, name)
                print(path)
                #exit()
                ### Remember to change indexes
                if len(path) == 36:
                    part1 = path[0:31]
                    part2 = '00'
                    part3 = path[31:]
                    new_name = '' + str(part1) + str(part2) + str(part3)
                    print("new_name:", new_name)
                    os.rename(path, new_name)
                if len(path) == 37:    
                    part1 = path[0:31]
                    part2 = '0'
                    part3 = path[31:]
                    new_name = '' + str(part1) + str(part2) + str(part3)
                    print(new_name)
                    os.rename(path, new_name)


def new_csv(label_path, desti):
    full_csv = []
    
    for root, dirs, files in os.walk(label_path, topdown=True):
        for name in files:
                line = str(name[:len(name)-3]) + 'jpg,' + name 
                full_csv.append(line)
                #000005.jpg,000005.txt
    
    X_train, X_test = train_test_split(full_csv, test_size=0.3, random_state=404)
    X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=404)

    pd.DataFrame(X_train).to_csv(desti + "cls2train.csv", header=False, index=False)
    pd.DataFrame(X_test).to_csv(desti + "cls2test.csv", header=False, index=False)
    pd.DataFrame(X_val).to_csv(desti + "cls2val.csv", header=False, index=False)




def change_classes(label_path, desti):
    for root, dirs, files in os.walk(label_path, topdown=True):
        for name in files:
                #print(name)
                path = os.path.join(root, name)
                file = open(path)
                lines = file.readlines()
                out_lines = []
                out_line = ''
                for line in lines:
                    if line[0:2] == '14':
                        out_line = str("0") + line[2:]
                        out_lines.append(out_line)
                    if line[0] == '6':
                        out_line = str("1") + line[1:]
                        out_lines.append(out_line)
                file.close()
                if len(out_line) == 0:
                    continue
                out_path = os.path.join(desti, name)
                file = open(out_path, 'w')
                for line in out_lines:
                    file.writelines(line)
                file.close()


    return 0


def change_file_type(img_path, desti):
    os.mkdir(desti)
    for root, dirs, files in os.walk(img_path, topdown=True):
        for name in files:
                path = os.path.join(root, name)
                img = Image.open(path)
                """
                print("path:")
                print(path)
                """
                ### Remember to change indexes
                if len(path) == 30:
                    part1 = desti
                    part11 = path[17:25]
                    part2 = '00'
                    part3 = path[25]
                    part4 = '.jpg'
                    new_name = '' + str(part1) + str(part11) + str(part2) + str(part3) + str(part4)
                    
                    #print(new_name)
                    img.save(new_name)
                if len(path) == 31:
                    part1 = desti
                    part11 = path[17:25]
                    part2 = '0'
                    part3 = path[25:27]
                    part4 = '.jpg'
                    new_name = '' + str(part1) + str(part11) + str(part2) + str(part3) + str(part4)
                    new_path = os.path.join(root, new_name)
                    #print(new_name)
                    img.save(new_name)
                if len(path) == 32:
                    part1 = desti
                    part11 = path[17:25]
                    part2 = ''
                    part3 = path[25:28]
                    part4 = '.jpg'
                    new_name = '' + str(part1) + str(part11) + str(part2) + str(part3) + str(part4)
                    new_path = os.path.join(root, new_name)
                    #print(new_name)
                    img.save(new_name)

def make_csv_dict(names, test_name):
    X_test = []
    X_train = []
    for i in range(len(names)):
        if names[i][:7] == test_name:
            X_test.append(names[i])
        else:
            X_train.append(names[i])
    train = []
    test_out = []
    #names[i][0:14] + ".txt"
    for t in X_train:
        train.append(t[:8] + t[11:14] + ".jpg," + t[0:14] + ".txt")
    for t in X_test:
        test_out.append(t[:8] + t[11:14] + ".jpg," + t[0:14] + ".txt")    
    
    train_out, val_out = train_test_split(train, test_size=0.15, random_state=404)

    pd.DataFrame(train_out).to_csv("LS_train_t.csv", header=False, index=False)
    pd.DataFrame(val_out).to_csv("LS_val_t.csv", header=False, index=False)
    pd.DataFrame(test_out).to_csv("LS_test_t.csv", header=False, index=False)


    return 1

def convert_bbox_corners(bboxes, scale):
    ret_bboxes = []
    print("input bboxes:", np.shape(bboxes))
    for i in range(len(bboxes)):
        temp = []
        for box in bboxes[i][0]:
            minscale =  1 - (scale - 1) 
            x1 = (box[1] - box[3] /2) * minscale
            y1 = (box[2] - box[4] /2) * minscale

            x2 = box[1] + box[3] /2
            y2 = box[2] + box[4] /2
            if (x2 * scale) < 1.0:
                x2 = x2 * scale
            if (y2 * scale) < 1.0:
                y2 = y2 * scale

            temp.append([[x1, y1], [x2, y2]])
        ret_bboxes.append(temp)
    return ret_bboxes

def crop_pic(bound_box, img):
    images = []
    for box in bound_box:
        h = np.shape(img)[0]
        w = np.shape(img)[1]
        x1 = math.floor(box[0][0]*w)
        x2 = math.floor(box[1][0]*w)
        y1 = math.floor(box[0][1]*h)
        y2 = math.floor(box[1][1]*h)
        crop = img[y1:y2, x1:x2] #11:1079, 96:948
        images.append(crop)

    return images

img_path = "LASIESTA/colFram"

label_path = img_path + '-GT'
print(label_path)

desti = img_path + "_jpg/"
print(desti)


#change_file_type(img_path, desti)

#change_names(label_path)

data_set, names = load_images_from_folder(label_path)

#for i in range(10):
#    print(names[i][:7] )

data_set_np = np.array(data_set)
print("shape of data:", np.shape(data_set_np))
bboxes = []
for i in range(len(data_set_np)):
    print("Algo1 nr.1 :", i ,"out of", len(data_set_np), end="\r")
    bboxes.append(alt_get_bounding_boxes_labels(data_set_np[i], '0'))

#for i in range(5):
#    print(bboxes[i+200][0])
#    print(np.shape(bboxes[i+200]))

O_bboxes = convert_bbox_corners(bboxes, 1.1)

print("O_bboxes shape", np.shape(O_bboxes))
data_set_img, names2 = load_images_from_folder(img_path)
crops = []
crops_GT = []

for i in range(len(O_bboxes)):
    print("O_bboxes :", i ,"out of", len(O_bboxes), end="\r")
    crops.append(crop_pic(O_bboxes[i], data_set_img[i]))
    crops_GT.append(crop_pic(O_bboxes[i], data_set[i]))

crops_GT_np = np.array(crops_GT)

s_crops = []
s_crops_GT = []


for i in range(len(crops)):
    print("Sqaure :", i ,"out of", len(crops), end="\r")
    temp1 = []
    temp2 = []
    for j in range(len(crops[i])):
        s_crop, b1 = Algo1.sqaure_crop(crops[i][j])
        s_crop_GT, b2 = Algo1.sqaure_crop(crops_GT[i][j])
        temp1.append(s_crop)
        temp2.append(s_crop_GT)
    s_crops.append(temp1)
    s_crops_GT.append(temp2)


"""
print(np.shape(s_crops))
print(np.shape(s_crops[0]))
print(np.shape(s_crops[0][0]))
for i in range(10, 1000, 10):
    for cro in s_crops_GT[i]:
        print(np.shape(cro))
        cro = cro.astype(np.uint8)
        cv2.imshow("Test", cro)
        cv2.waitKey(0)
"""

s_crop_GT_np = np.array(s_crops_GT)
print("shape of s_crop_GT", np.shape(s_crops_GT))
crop_bboxes = []
for i in range(len(s_crops_GT)):
    temp = []
    print("Algo1 nr.2:", i ,"out of", len(s_crops_GT), end="\r")
    for s_c in s_crops_GT[i]:
        try:
            s_c = np.array(s_c)
            s_c = s_c.astype(np.uint8)
        except:
            print(np.shape(s_c))
            print("i:", i)
            print("fuck you")
        temp.append(alt_get_bounding_boxes_labels(s_c, '0'))
    crop_bboxes.append(temp)


save_labels_name = img_path + '_crop_labels'
os.mkdir(save_labels_name)
for i in range(len(names)):
    save_bboxes_txt(crop_bboxes[i], names[i], save_labels_name)


print(np.shape(crops))
print(np.shape(crops[0]))
print(np.shape(crops[0][0]))
for i in range(10, 1000, 10):
    for cro in s_crops[i]:
        print(np.shape(cro))
        cv2.imshow("Test", cro)
        cv2.waitKey(0)



exit()


save_labels_name = img_path + '_labels'
save_bboxes_txt(bboxes, names, save_labels_name)

test_name = 'O_CL_02'
make_csv_dict(names, test_name)

