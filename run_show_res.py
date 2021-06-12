from pathlib import WindowsPath
from typing import Counter
import numpy as np
from numpy.lib import utils
import Algo1
import predictCnn
import torch
import torchvision.transforms
import cv2 as cv2
import matplotlib.pyplot as plt
import coarseGrained
import cv2
import config
import utils
import time
from scipy.ndimage import gaussian_filter



def CNN(model, cropped_frames, Other_bounding_box, CONF_THRESHOLD, NMS_IOU_THRESH, ori_frame_shape):
    bboxes = []

    for idx in range(len(cropped_frames)):
        c_fra = cropped_frames[idx]

        input_size = 416

        #Rezising of the input frame to fit the model
        if (np.shape(c_fra)[0] < input_size) and (np.shape(c_fra)[1] < input_size):
            pil_frame, sqaure_shape = Algo1.size_up(c_fra, input_size)
        else:
            pil_frame = cv2.resize(c_fra, (416,416))
        pil_frame = pil_frame.astype(np.uint8)

        #Uncomment to show input for the model
        """cv2.imshow("Input frame", fra)
        k = cv2.waitKey(3) & 0xFF
        if k == 27:
            break
        q = input("Pres enter to continue, n for next model, q to exit")
        if (q == "q"):
            cv2.destroyAllWindows()
            exit()
        if (q == "n"):
            break"""

        ANCHORS = config.ANCHORS
        pil_fra_min = np.min(pil_frame)
        pil_fra_max = np.max(pil_frame)
        pil_fra = torch.from_numpy(np.array([pil_frame]))
        pil_fra = pil_fra.permute(0,3,1,2)
        pil_fra = pil_fra.float()
        
        pil_fra = (pil_fra - pil_fra_min) / (pil_fra_max - pil_fra_min)

        var = predictCnn.get_bboxes(
            pil_fra,
            model,
            NMS_IOU_THRESH, 
            ANCHORS,
            CONF_THRESHOLD
        )
        
        #Uncomment to show preds on cropped frames 
        #draw_boxes(var[0], pil_frame_crop)
        
        if np.shape(var) == (1,0):
            continue
        else:
            bboxes.append(Algo1.scale_bboxes(var[0], ori_frame_shape, Other_bounding_box[idx], sqaure_shape))
        
    return bboxes


def main():

    paths = ['test_frames/test_I_IL_01', 'test_frames/test_I_SI_01', 'test_frames/test_O_OP_01', 
            'test_frames/test_O_CM_01', 'test_frames/test_O_OF_01']
    data_sets = []
    for p in paths:
        data_sets.append(utils.load_images_from_folder(p))

    paths_label = ['test_labels/label_I_IL_01', 'test_labels/label_I_SI_01', 'test_labels/label_O_OP_01', 
                    'test_labels/label_O_CM_01', 'test_labels/label_O_OF_01']
    GTs = []
    for p in paths_label:
        GTs.append(utils.load_labels(p))
    
    gt_shows = []
    for p in paths_label:
        gt_shows.append(utils.load_labels_test(p))
    
    maps = []
    f_measures = []

    total_time = []
    Coarse_time = []
    Algo1_time = []
    CNN_time = []
    
    for i in range(len(data_sets)):
        data_set = data_sets[i]
        GT = GTs[i]
        gt_show = gt_shows[i]
        model = 'model_weights.h5'
        model = torch.load(model, map_location=torch.device('cuda'))
        model.eval()
        count = 0
        col_bboxes = []
        while (len(data_set)-1):
            t0 = time.time()
            print("While loop :", count ,"out of", len(data_set), end="\r")
            if(count == len(data_set) -1):
                break

            old_frame = data_set[count]
            new_frame = data_set[count+1]
            count += 1

            ori_frame_shape = np.shape(new_frame)
            #if ret == False:
            #   break
            # Block 1: Coarse grained
            gauss_sigma = 5
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            old_frame = gaussian_filter(old_frame, sigma=gauss_sigma)
            c_grained, n_fram = coarseGrained.dense_optical_flow(cv2.calcOpticalFlowFarneback,
                        gauss_sigma, 5, new_frame, old_frame,
                        [0.5, 20, 50, 7, 20, 1.1, 0], True, True) 
            
            #Show output of Coarse Grained
            """cv2.imshow("Original frame", c_grained)
            k = cv2.waitKey(3) & 0xFF
            if k == 27:
                break
            q = input("Pres enter to continue, n for next model, q to exit")
            if (q == "q"):
                cv2.destroyAllWindows()
                exit()
            if (q == "n"):
                break"""

            t1 = time.time()
            # Block 2: Algorithm 1
            pixel_thresh = 15
            Bounding_box = Algo1.Algorithm1(c_grained, pixel_thresh)
            #print("bbox",Bounding_box)

            Other_bounding_box = Bounding_box
            
            Other_bounding_box = []
            for i in range(len(Bounding_box)):
                if (Bounding_box[i][1][0]-Bounding_box[i][0][0])*(Bounding_box[i][1][1]-Bounding_box[i][0][1]) > 2000:
                    Other_bounding_box.append(Bounding_box[i])
            
            if len(Other_bounding_box) < 1:
                continue
            else:

                cropped_frames = utils.crop_pic(Other_bounding_box, new_frame)

                # Block 3: CNN object recognition
                NMS_IOU_THRESH = 0.45
                CONF_THRESHOLD = 0.05
                t2 = time.time()
                bboxes = CNN(model, cropped_frames, Other_bounding_box, CONF_THRESHOLD, NMS_IOU_THRESH, ori_frame_shape)

                #### Present results for each frame ####

                #Uncomment the following line to see the prediction on the frame
                utils.draw_boxes(bboxes, new_frame)

                #Uncomment the following line to see the label on the frame
                #utils.draw_boxes(gt_show[count - 1], new_frame)
                save_bbox = bboxes.copy()
                for box in save_bbox:
                    col_bboxes.append([count] + box)
                t3 = time.time()
            
            total_time.append(t3 - t0)
            Coarse_time.append(t1 - t0)
            Algo1_time.append(t2 - t1)
            CNN_time.append(t3 - t2)


        #Calculate mAP and f-measure
        
        Map, f_measure = utils.map_pred(col_bboxes, GT)
        Map = Map.cpu().detach().numpy()
        f_measure = f_measure.cpu().detach().numpy()
        
        f_measures.append(f_measure)
        maps.append(Map)
        print("mAP value:", Map)
        print("f-measure value:", f_measure)
        

    print("Average time per frame:")
    print(sum(total_time) / len(total_time))
    print("Avg. Coarse grained time:")
    print(sum(Coarse_time) / len(Coarse_time))
    print("Avg. Algo1 time:")
    print(sum(Algo1_time) / len(Algo1_time))
    print("Avg. CNN time:")
    print(sum(CNN_time) / len(CNN_time))



if __name__ == '__main__':
    main()

