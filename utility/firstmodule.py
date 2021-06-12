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

def load_frames(file):
    count = 0
    videoFile = file
    cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    x = 1
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = "frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print("Done!")


# Frame difference with math morphology and filtering

def morph(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

# Gray scale function without using CV2 lib.
def gray_scaling(image):
    gray = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07])
    return gray(image)

# Low pass filter not in use right now
def low_pass_filter(image):

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)  # save image of the image in the fourier domain.

    # plot both images

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back


# convert a image to binary using Otsu's algorithm
def convert_to_bin(image):
    tresh, binary = cv2.threshold(np.float32(image),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return tresh, binary


def crop_pic(bound_box, img):
    images = []
    for box in bound_box:
        crop = img[box[0][1]:box[1][1], box[0][0]:box[1][0]] #11:1079, 96:948
        images.append(crop)

    return images

def draw_boxes(bboxes, image):
    PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
    ]
    print("shape of bboxes:", np.shape(bboxes))
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(PASCAL_CLASSES))]
    shape = np.shape(image)
    bad_runs_counter = 0
    #image = image[0]
    #image = np.transpose(image, (0,1,2))
    print(np.shape(image))
    pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #pyplot.imshow(image)
    ax = pyplot.gca()

    for box in bboxes:
        class_pred = box[0]
        pred_name = PASCAL_CLASSES[int(class_pred)]
        prob = round(box[1], 2)
        predict = PASCAL_CLASSES[int(class_pred)]
        x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
        if (x1 > 1 or y1 > 1 or x2  > 1 or y2 > 1):
            bad_runs_counter += 1
            #print("x or y coordinate not good, bad runs: ", bad_runs_counter)
            #continue
        """
        x1 = x1 * shape[0]
        x2 = x2 * shape[0]
        y1 = y1 * shape[1]
        y2 = y2 * shape[1]
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='black')
        pyplot.text(x1, y1, label, color='black')
        """
        im = np.array(image)
        height, width, _ = im.shape
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        #print(upper_left_x)
        #print(upper_left_y)
        #print(upper_left_x * width, upper_left_y * height)
        rect = Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        pyplot.text(
            upper_left_x * width,
            upper_left_y * height,
            s=(pred_name, " prob:", prob),
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

        
        ax.add_patch(rect)
        
        label = "%s (%.3f)" % (predict, prob)
    print("x or y coordinate not good, bad runs: ", bad_runs_counter)
    pyplot.show()

# https://learnopencv.com/optical-flow-in-opencv/
def scale_bboxes(bbox, frame_box, s_bboxes, ori_frame_shape):
    #Scales bounding boxes to original frame from 
    #bounding boxes given by model from cropped frame
    scaled_bboxes = s_bboxes
    for box in bbox:
        x1, y1, x2, y2 = box[2], box[3], box[4], box[5]
        ori_y2, ori_x2, channels = ori_frame_shape
        x1_n = ((x1 * (frame_box[1][0] - frame_box[0][0])) + (frame_box[0][0])) / ori_x2
        x2_n = ((x2 * (frame_box[1][0] - frame_box[0][0])) + (frame_box[0][0])) / ori_x2
        y1_n = ((y1 * (frame_box[1][1] - frame_box[0][1])) + (frame_box[0][1])) / ori_y2
        y2_n = ((y2 * (frame_box[1][1] - frame_box[0][1])) + (frame_box[0][1])) / ori_y2
        box[2], box[3], box[4], box[5] = x1_n, y1_n, x2_n, y2_n 
        scaled_bboxes.append(box)

    return scaled_bboxes

def dense_optical_flow(method, gauss_sigma, kernel_size, video_path, params=[], filtering = True, morphing = True):
    # Read the video and first frame
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    all_frame_box = []
    final_array = []
    model = 0
    count = 0

    
    #gray scaling and low density filtering of next frame
    if filtering:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # old_frame = low_pass_filter(old_frame)
        old_frame = gaussian_filter(old_frame, sigma=gauss_sigma)

    while count <10:
        # Read the next frame
        ret, new_frame = cap.read()
        ori_frame = new_frame
        # frame_copy = new_frame
        ori_frame_shape = np.shape(ori_frame)
        print("ori frame shape", ori_frame_shape)
        if not ret:
            break

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
            norm_flow = morph(norm_flow, kernel_size)

        pixel_thresh = 100 # minimum of pixel for sides of bounding boxes
        t0 = time.time()
        Bounding_box = Algo1.Algorithm1(norm_flow, pixel_thresh)
        t1 = time.time()
        print("B-box time:" + str(t1-t0))
        t0 = 0
        t1 = 0


        filename = "frame%d.png" % count
        count += 1
        final_array.append([norm_flow, count])
        print("frame number", count)


        Other_bounding_box = []
        for i in range(len(Bounding_box)):
            if (Bounding_box[i][1][0]-Bounding_box[i][0][0])*(Bounding_box[i][1][1]-Bounding_box[i][0][1]) > 100000:
                Other_bounding_box.append(Bounding_box[i])

        if (model == 0):
            model = torch.load('models2/model2_e_80.h5')
        
        cropped_frames = crop_pic(Other_bounding_box, ori_frame)

        bboxes = []

        for idx in range(len(cropped_frames)):
            fra = cropped_frames[idx]
            #cv2.imshow("Original frame", fra)
            
            #fra = cv2.imread('khani_test.jpg') #if want to test for a image
            """
            k = cv2.waitKey(3) & 0xFF
            if k == 27:
                break
            q = input("Pres enter to continue, n for next model, q to exit")
            if (q == "q"):
                cv2.destroyAllWindows()
                exit()
            if (q == "n"):
                break
            """
            print("image pre resize:", np.shape(fra))
            pil_frame = cv2.resize(fra, (416,416))
            pil_fra = np.array(Image.fromarray(pil_frame).convert("RGB"))
            print(pil_fra.shape)
            NMS_IOU_THRESH = 0.4
            ANCHORS =[
                [(0.28,0.22),(0.38,0.48),(0.9,0.78)],
                [(0.07,0.15),(0.15,0.11),(0.14,0.29)],
                [(0.02,0.03),(0.04,0.07),(0.08,0.06)]
            ]

            pil_fra = torch.from_numpy(np.array([pil_frame]))
            pil_fra = pil_fra.permute(0,3,1,2)
            pil_fra = pil_fra.float()
            print(pil_fra.shape)
            pil_fra /= 255

            CONF_THRESHOLD = 0.75
            var = predictCnn.get_bboxes(
                pil_fra,
                model,
                NMS_IOU_THRESH, 
                ANCHORS,
                CONF_THRESHOLD
            )   
            print(np.shape(var))

            bboxes = scale_bboxes(var[0], Other_bounding_box[idx], bboxes, ori_frame_shape)
            
    
        draw_boxes(bboxes, ori_frame)  

        all_frame_box.append(Other_bounding_box)
        old_frame = new_frame

    return all_frame_box



# Gives the binary representation of each frame from the video in the call. Takes the optical flow method from CV2,
# the video, some array of params (from web side) and two booleans on for filtering one for morphology.
result = dense_optical_flow(cv2.calcOpticalFlowFarneback,0.5, 5, "running.mp4", [0.5, 5, 15, 5, 5, 1.1, 0], True, True)


print("we got out")



