# follow the ctrl command
import os
import cv2
import time
import math
import pickle
import numpy as np
from pyquaternion import Quaternion
from video_control import StreamingExample


import olympe
from video_control import StreamingExample
from keyboard_ctrl import KeyboardCtrl
from pynput.keyboard import Listener, Key, KeyCode
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.enums.ardrone3.PilotingState import FlyingStateChanged_State
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD

CtrlTime = 0.02
image_difference_threshold = 0.70
orient_difference_threshold = 0.005
height_difference_threshold = 0.015


def create_dir(file_dir_list):
    for file_dir in file_dir_list:
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

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


Data_root = './data'
Output_root = './train_data_complete'
create_dir(['./train_data_complete', './train_data_complete/image', './train_data_complete/meta', './train_data_complete/ctrl'])
Ctrl_seq = np.loadtxt('./data/ctrl_seq.txt').astype(int)
Checkpoint = np.loadtxt('./data/checkpoint.txt').astype(int)
# load image and meta at checkpoint
Images = {}
Metadata = {}
for ind in Checkpoint:
    img = cv2.imread(os.path.join(Data_root,'image',str(ind)+'.png'))
    kp1, des1 = orb.detectAndCompute(img, None)
    Images[ind] = des1
    meta = np.loadtxt(os.path.join(Data_root,'metadata',str(ind)+'.txt'))
    Metadata[ind] = meta



def fly(drone, streaming_example):
    drone.connection()
    drone.start_piloting()
    control = KeyboardCtrl()
    for corr_count in range(10):
        while True:
            if control.takeoff():
                drone(TakeOff() & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait() 
                break
        time.sleep(2.5)
        cpind = 0
        for i in range(Ctrl_seq.shape[0]):
            cur_count = Ctrl_seq[i, 0]
            ctrl = Ctrl_seq[i, 1:]
            print(Ctrl_seq[i,:])
            drone.piloting_pcmd(ctrl[0], ctrl[1], ctrl[2], ctrl[3], CtrlTime)
            time.sleep(CtrlTime)
            if cur_count == Checkpoint[cpind]:
                cmd_img_dir = os.path.join('./train_data_complete/image', str(cur_count))
                cmd_dict_dir = os.path.join('./train_data_complete/meta', str(cur_count))
                cmd_ctrl_dir = os.path.join('./train_data_complete/ctrl', str(cur_count))
                create_dir([cmd_img_dir, cmd_dict_dir, cmd_ctrl_dir])
                img = streaming_example.current_frame
                meta = read_meta(streaming_example.meta_other)
                data_diff = diff(img, meta, Images[cur_count], Metadata[cur_count])
                np.savetxt(os.path.join(cmd_ctrl_dir, str(corr_count)+'data_diff.txt'),np.array(data_diff), fmt='%1.4f', newline='\n')
                # correction
                ctrl_seq = []
                if np.abs(data_diff[1]) > 1.5:
                    ori_const = data_diff[1]
                else:
                    ori_const = 0
                print("The diff before correction: ", [data_diff[0], data_diff[1]-ori_const, data_diff[2]])
                if data_diff[0] > image_difference_threshold and abs(data_diff[1]-ori_const) < orient_difference_threshold and abs(data_diff[2]) < height_difference_threshold:
                    np.savetxt(os.path.join(cmd_ctrl_dir, str(corr_count)+'ctrl.txt'),np.array(ctrl_seq))
                    cpind += 1
                    continue
                while True:
                    if control.break_loop():
                        print("Finish correction")
                        cv2.imwrite(os.path.join(cmd_img_dir, str(corr_count)+'after_cor.png'), img)
                        np.savetxt(os.path.join(cmd_dict_dir, str(corr_count)+'after_cor.txt'),np.array(meta), fmt='%1.4f', newline='\n')
                        break
                    elif control.takeoff():
                        drone(TakeOff(_no_expect=True) & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                    elif control.landing():
                        drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
                    elif control.has_piloting_cmd():   # correction
                        img = streaming_example.current_frame
                        meta = read_meta(streaming_example.meta_other)
                        data_diff = diff(img, meta, Images[cur_count], Metadata[cur_count])
                        print("diff: ", data_diff)
                        drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), CtrlTime)
                        time.sleep(CtrlTime)
                        data_diff.extend([control.roll(), control.pitch(), control.yaw(), control.throttle()])
                        ctrl_seq.append(data_diff)
                        if data_diff[0] > image_difference_threshold and abs(data_diff[1]) < orient_difference_threshold and abs(data_diff[2]) < height_difference_threshold: 
                            print("Finish correction")
                            cv2.imwrite(os.path.join(cmd_img_dir, str(corr_count)+'after_cor.png'), img)
                            np.savetxt(os.path.join(cmd_dict_dir, str(corr_count)+'after_cor.txt'),np.array(meta), fmt='%1.4f', newline='\n')
                            break
                np.savetxt(os.path.join(cmd_ctrl_dir, str(corr_count)+'ctrl.txt'),np.array(ctrl_seq), fmt='%1.4f %1.4f %1.4f %d %d %d %d', newline='\n')
                cpind += 1
        drone(
            Landing()
            >> FlyingStateChanged(state="landed", _timeout=5)
        ).wait()



if __name__ == "__main__":
    with olympe.Drone("192.168.42.1") as drone:
        streaming_example = StreamingExample(drone, True)
        # Start the video stream
        streaming_example.start()
        
        fly(drone, streaming_example)
        
        # Stop the video stream
        streaming_example.stop()
        # Recorded video stream postprocessing
        streaming_example.postprocessing()
