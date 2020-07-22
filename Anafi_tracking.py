from __future__ import print_function
import cv2
import numpy as np
import imutils
import os
import pickle
import time
from keyboard_ctrl import KeyboardCtrl
import pika
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import copy


'''

cv2 coordinate system (x,y):
----x(c)----->
|
|
y(r)
|
+

The room
+
|
y
|
|door-------x------->

The camera frame
           |
           |
-------------------x-->
           |
           y
           |
           +
'''

SAVEFIG = False
PLOT = True
FIG_SAVE_PATH = './dataset_img'
if SAVEFIG:
    if not os.path.isdir(FIG_SAVE_PATH):
        os.mkdir(FIG_SAVE_PATH)
SAVEVID = True
weights = './models/best.pt'
imgsz = 640

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

# channel.queue_declare(queue='hello')
channel.exchange_declare(exchange='logs',
                         exchange_type='fanout')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # only has 1 GPU set GPU 0 as default
half = device != 'cpu'
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
if half:
    model.half()  # to FP16

def detect_yolo(img0, PLOT = True):
    imgsz = 640
    # Load image
    img = letterbox(img0, new_shape=imgsz)[0]     # Padded resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)  # change to c-contingous memory storage to speed up
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)     # Apply NMS

    # Rescale boxes from img_size to img0 size
    if pred[0] is not None and len(pred):
        det = pred[0]
        xyxy = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  # (x1, y1) (x2, y2)
        center = np.array((int((xyxy[0][0] + xyxy[0][2])/2), int((xyxy[0][1] + xyxy[0][3])/2)))
        plot_one_box(xyxy[0], img0, line_thickness=3)
        return center, img0
    else:
        return None

def detect_bgs(frame):
    frame = copy.copy(frame)
    fgMask = backSub.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fgMask = cv2.erode(fgMask,kernel,iterations = 1)
    fgMask = cv2.dilate(fgMask,kernel,iterations = 1)

    cnts = cv2.findContours(fgMask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    # if empty just ignore
    if cnts == []:
        return None
    else:
        area_cnts = [] # find the largest area
        for ind, c in enumerate(cnts):
           area = cv2.contourArea(c)
           area_cnts.append(area)
        mind = area_cnts.index(max(area_cnts))
        cmax = cnts[mind]
        cmax_list.append(cmax)  # store the largest information
        if area_cnts[mind] > 250:  # only keep area larger than 200
           x, y, w, h = cv2.boundingRect(cmax)
           cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
           center = (int(x+0.5*w), int(y+0.5*h))

        else:  # smaller area is ignored and replace by its previous rect
           x, y, w, h = cv2.boundingRect(cmax_list[-1])
           cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
           center = (int(x+0.5*w), int(y+0.5*h))
    return center, frame

with open('./camera_calibration_API/results.pickle', 'rb') as f:
    cam_dic = pickle.load(f)

def undistort_img(img):
    mtx = cam_dic["intrinsic_matrix"]
    dist = cam_dic["distortion_coefficients"]
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
    


cap = cv2.VideoCapture(0)


## [create]
backSub = cv2.createBackgroundSubtractorMOG2()

## [create]

cmax_list = []
center_list = []

if SAVEVID:
    out = cv2.VideoWriter('./follow_path.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))

curcount = 0
control = KeyboardCtrl()
savecount = 0
nodetect = False
while(1):
    # Take each frame
    ret, frame = cap.read()
    if frame is None:
        break 

    frame = undistort_img(frame)   # normal undistortion not fisheye
    if (SAVEFIG and (curcount % 20 == 0)) or (center_list and np.linalg.norm(np.array(center) - np.array(center_list[-1])) > 80):
        savecount += 1
        cv2.imwrite(FIG_SAVE_PATH + '/' + str(savecount) + '.png', frame)
    elif SAVEFIG and control.checkpoint():
        savecount += 1
        cv2.imwrite(FIG_SAVE_PATH + '/' + str(savecount) + '.png', frame)
    
    center_frame_yolo = detect_yolo(frame)
    center_frame_bgs = detect_bgs(frame)
    if center_frame_yolo:
        center, frame = center_frame_yolo
    elif center_frame_bgs:
        center, frame = center_frame_bgs
    else:
        nodetect = True
    # Plot bbox
    if PLOT:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    if control.quit():
        np.save("./center_list.npy", np.array(center_list))
        cap.release()
        if SAVEVID:
            out.release()
        cv2.destroyAllWindows()
        connection.close()
        break

    if nodetect:
        continue
    if curcount > 10:
        print(center)
        center_list.append(center)
        centerstr = str(center[0]) + ' ' + str(center[1])
        channel.basic_publish(exchange='logs', routing_key='hello', body=centerstr)
    if SAVEVID:
        out.write(frame)
    curcount += 1 

# draw picture
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.figure()
# ax = fig.add_subplot(1, 1, 1)
center_list = np.array(center_list)
# convert the rc -> xy
center_list[:, 1] = 480 - center_list[:, 1]
plt.plot(center_list[:,0], center_list[:,1], 'o-')
plt.show()


