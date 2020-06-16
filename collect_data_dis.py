import os
import cv2
import time
import math
import pickle
import numpy as np
from pyquaternion import Quaternion
from video_control import StreamingExample

import olympe
from keyboard_ctrl import KeyboardCtrl
from pynput.keyboard import Listener, Key, KeyCode
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD


Data_output = './train_data'
Correctimestamp = 0.02
image_difference_threshold = 0.70
orient_difference_threshold = 0.005
height_difference_threshold = 0.005


# take off

# save the standard/normal image

# slightly modifying/waiting for drifting

# record current frame and correction command

def read_meta(metadata):
    w = metadata['drone_quat']['w']
    x = metadata['drone_quat']['x']
    y = metadata['drone_quat']['y']
    z = metadata['drone_quat']['z']
    g_d = metadata['ground_distance']
    exp_meta = [w,x,y,z,g_d]
    return exp_meta

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def diff(img, meta, des1, exp_meta):

    kp2, des2 = orb.detectAndCompute(img,None)
    # Match descriptors.
    matches = len(bf.match(des1,des2))
    
    # quaternion norm
    q_meta = Quaternion(array=np.array(meta[:4]))
    q_exp = Quaternion(array=np.array(exp_meta[:4]))
    q_diff = q_exp.angle - q_meta.angle   # positive if bias to left, negative if bias to right


    # height difference
    h_diff = meta[-1] - exp_meta[-1]
    

    return [matches/len(des1), q_diff, h_diff] #

def fly(drone, streaming_example):
    drone.connection()
    drone.start_piloting()
    control = KeyboardCtrl()
    count = -1
    while not control.quit():
        count += 1 # num of trajectory recorded
        # create new directory
        img_dir = os.path.join(Data_output, 'image', str(count))
        dict_dir = os.path.join(Data_output, 'metadata', str(count))
        ctrl_dir = os.path.join(Data_output, 'ctrl', str(count))
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        if not os.path.isdir(dict_dir):
            os.mkdir(dict_dir)
        if not os.path.isdir(ctrl_dir):
            os.mkdir(ctrl_dir)

        while True:
            if control.takeoff():
                drone(TakeOff(_no_expect=True) & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
            elif control.break_loop():
                img_exp = streaming_example.current_frame
                meta_exp = read_meta(streaming_example.meta_other)
                cv2.imwrite(os.path.join(img_dir, 'exp_img.png'), img_exp)
                np.savetxt(os.path.join(dict_dir, 'exp_meta.txt'),np.array(meta_exp), fmt='%1.4f', newline='\n')
                kp1, des1 = orb.detectAndCompute(img_exp,None)
                break
        
        
        corr_count = 6
        while not control.break_loop():
            if control.takeoff():

                drone(TakeOff())
            elif control.landing():
                drone(Landing())                            
            elif control.has_piloting_cmd():  # add error
                drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), Correctimestamp)
                time.sleep(Correctimestamp)
            elif control.wait():  # save the error state/img, press space
                corr_count += 1
                img_err = streaming_example.current_frame
                meta_err = read_meta(streaming_example.meta_other)
                cv2.imwrite(os.path.join(img_dir, str(corr_count)+'error.png'), img_err)
                np.savetxt(os.path.join(dict_dir, str(corr_count)+'error.txt'),np.array(meta_err), fmt='%1.4f', newline='\n')
                # calculate error match
                kp0, des0 = orb.detectAndCompute(img_err,None)
                match_err = len(bf.match(des1,des0))
                print("Error img/state saved. Begin correction")
                ctrl_seq = []
                while True:
                    if control.break_loop():
                        print("Finish correction")
                        break
                    if control.has_piloting_cmd():   # correction
                        img = streaming_example.current_frame
                        meta = read_meta(streaming_example.meta_other)
                        data_diff = diff(img, meta, des1, meta_exp)
                        print("diff: ", data_diff)
                        if data_diff[0] > image_difference_threshold and abs(data_diff[1]) < orient_difference_threshold and abs(data_diff[2]) < height_difference_threshold: 
                            print("Finish correction")
                            cv2.imwrite(os.path.join(img_dir, str(corr_count)+'after_cor.png'), img)
                            np.savetxt(os.path.join(dict_dir, str(corr_count)+'after_cor.txt'),np.array(meta), fmt='%1.4f', newline='\n')
                            break
                        drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), Correctimestamp)
                        time.sleep(Correctimestamp)
                        data_diff.extend([control.roll(), control.pitch(), control.yaw(), control.throttle()])
                        ctrl_seq.append(data_diff)
                np.savetxt(os.path.join(ctrl_dir, str(corr_count)+'ctrl.txt'),np.array(ctrl_seq), fmt='%1.4f %1.4f %1.4f %d %d %d %d', newline='\n')
        drone(
            Landing()
            >> FlyingStateChanged(state="landed", _timeout=5)
        ).wait()

if __name__ == "__main__":
    with olympe.Drone("192.168.42.1") as drone:
        streaming_example = StreamingExample(drone, True)
        streaming_example.datafile = Data_output
        # Start the video stream
        streaming_example.start()
        
        fly(drone, streaming_example)
        
        # Stop the video stream
        streaming_example.stop()
        # Recorded video stream postprocessing
        streaming_example.postprocessing()
'''
descript = cv2.xfeatures2d.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
orb = cv.ORB_create()
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
def diff(img, meta, des1, exp_meta): 
    # ratio of detected same SIFT features
    kp2, des2 = descript.detectAndCompute(img,None)
    matches = flann.knnMatch(des1,des2,k=2)

    count1 = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            count1+=1
'''