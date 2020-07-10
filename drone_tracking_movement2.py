from __future__ import print_function
import cv2
import numpy as np
import imutils
import os



'''

cv2 coordinate system (x,y):
----x----->
|
|
y
|
|

A star planning coordinate system (x,y):
^
|
|
y
|
|
----x---->

'''
def draw_path(frame):
    if os.path.exists('path_file_x.txt'):
        x_coord = []
        y_coord = []
        # open path file and read the content into a list
        with open('path_file_x.txt', 'r') as file_x:
            for line in file_x:
                currentPlace = int(line[:-1])
                x_coord.append(currentPlace)
        
        with open('path_file_y.txt', 'r') as file_y:
            for line in file_y:
                currentPlace = int(line[:-1])
                y_coord.append(currentPlace)
        height = frame.shape[0]
        width = frame.shape[1]
        for i in range(len(x_coord)-1):
            cv2.line(frame,(x_coord[i],height-y_coord[i]), (x_coord[i+1],height-y_coord[i+1]), (255,0,0), 3)
    else:
        pass


def undistort(img, K, D, DIM):
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

# def upright2downleft(center):
    

if __name__ == '__main__':


    cap = cv2.VideoCapture(1)

    #fisheye undistortion parameters
    cali = np.load('cali.npz')
    DIM = tuple(cali['DIM'])
    K = cali['K']
    D = cali['D']



    ## [create]
    backSub = cv2.createBackgroundSubtractorMOG2()

    ## [create]

    cmax_list = []
    center_list = []

    out = cv2.VideoWriter('close_demo.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (1280, 720))

    #clear the current location files
    open('location_x.txt', 'w').close()
    open('location_y.txt', 'w').close()

    binary1 = cv2.imread('binary_map.png')
    # binary1 = cv2.cvtColor(binary1, cv2.COLOR_BGR2GRAY)

    while(1):
        # Take each frame
        ret, frame = cap.read()
        if frame is None:
            break 

        frame = undistort(frame, K, D, DIM)

        fgMask = backSub.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.erode(fgMask,kernel,iterations = 1)
        fgMask = cv2.dilate(fgMask,kernel,iterations = 1)

        cnts = cv2.findContours(fgMask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

        # if empty just ignore
        if cnts == []:
            continue
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

            # if len(center_list) > 10:
            #     average = np.mean(np.array(center_list[-9:]), 0)
            #     if np.linalg.norm(average - np.array(center)) > 200:
            #         center_list.append(center_list[-1])
            #     else:
            #         center_list.append(center)
            # else:
            center_list.append(center)


            #write current location to txt file
            with open('location_x.txt', 'a+') as file_x:
                file_x.write('%i\n' % center[0])

            with open('location_y.txt', 'a+') as file_y:
                file_y.write('%i\n' % center[1])


            if len(center_list)>3:
                for i in range(3,len(center_list)):
                    cv2.line(frame, center_list[i], center_list[i-1], (0, 0, 255), 3)

            draw_path(frame)

        # out.write(frame)

        # plot the block
        binary2 = np.where(binary1>0.5, 0, 1)
        frame = (binary2 * frame).astype(np.uint8)

        cv2.imshow('input', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    out.release()

    cap.release()
    cv2.destroyAllWindows()