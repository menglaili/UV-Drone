# follow the ctrl command
import os
import cv2
import time
import math
import pika
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from multiprocessing import Pool
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
# image_difference_threshold = 0.70
orient_difference_threshold = 0.015
xyz_difference_threshold = 0.02
xyz_norm = 0.002
Correction_time = 2


with open("./camera_calibration_API/camera_mtx.pickle", 'rb') as f:
    camera_mtx = pickle.load(f)
camera_mtx = np.linalg.inv(camera_mtx)  #INVERSED
def find_xyz(center, z):
    '''
    camera_mtx: Instrinct parameter INVERSED
    center: camera pixel position of detected drone
    z: The height generate back from the drone
    Return: the location of drone in Camera Frame !!!
    '''
    z = 2.86 - z
    center = np.array(center)
    center = np.hstack((center, np.array([1])))
    image_frame = camera_mtx.dot(center)
    camera_frame = image_frame * z
    # camera_frame[2] = 2.86 - camera_frame[2]
    return camera_frame


connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='hello')

CAMERA_LIST = []
class ConsumerThread(threading.Thread):
    def __init__(self, host, *args, **kwargs):
        super(ConsumerThread, self).__init__(*args, **kwargs)

        self._host = host

    # Not necessarily a method.
    def callback_func(self, channel, method, properties, body):
        CAMERA_LIST.append(body) 

    def run(self):

        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self._host)) 

        channel = connection.channel()

        channel.exchange_declare(exchange='logs', exchange_type='fanout')

        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        channel.queue_bind(exchange='logs', queue=queue_name)

        
        channel.basic_consume(
            queue=queue_name, on_message_callback=self.callback_func, auto_ack=True)


        channel.start_consuming()


def create_dir(file_dir_list):
    for file_dir in file_dir_list:
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)

def read_meta(metadata, centerstr):
    w = metadata['drone_quat']['w']
    x = metadata['drone_quat']['x']
    y = metadata['drone_quat']['y']
    z = metadata['drone_quat']['z']
    g_d = metadata['ground_distance']
    camera_frame = find_xyz([int(centerstr[0]), int(centerstr[1])], g_d)
    exp_meta = [camera_frame[0],camera_frame[1],camera_frame[2],w,x,y,z]
    return np.array(exp_meta)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def diff(meta, exp_meta):
    
    # quaternion norm
    q_meta = Quaternion(array=np.array(meta[3:]))
    q_exp = Quaternion(array=np.array(exp_meta[3:]))
    q_diff = Quaternion.absolute_distance(q_exp, q_meta)   # positive if bias to left, negative if bias to right

    # x,y,z difference
    xyz_diff = meta[:3] - exp_meta[:3]
    

    return [xyz_diff, q_diff] #



Data_root = './data_new'
Output_root = './train_data_comp_new'
create_dir(['./train_data_comp_new', './train_data_comp_new/image', './train_data_comp_new/meta', './train_data_comp_new/ctrl'])
Ctrl_seq = np.loadtxt('./data_new/ctrl_seq.txt').astype(int)
Checkpoint = np.loadtxt('./data_new/checkpoint.txt').astype(int)
# load image and meta at checkpoint
Images = {}
Metadata = {}
for ind in Checkpoint:
    img = cv2.imread(os.path.join(Data_root,'image',str(ind)+'.png'))
    kp1, des1 = orb.detectAndCompute(img, None)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Images[ind] = [des1, img1]
    meta = np.loadtxt(os.path.join(Data_root,'metadata',str(ind)+'.txt'))
    Metadata[ind] = meta


def fly(drone, streaming_example):
    drone.connection()
    drone.start_piloting()
    control = KeyboardCtrl()
    for corr_count in range(1, Correction_time):
        while True:
            if control.takeoff():
                takeoffsign = True
                drone(TakeOff() & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait() 
                break
        time.sleep(2.5)
        cpind = 0
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        pose_list = []
        for i in range(Ctrl_seq.shape[0]):
            cur_count = Ctrl_seq[i, 0]
            ctrl = Ctrl_seq[i, 1:]
            print(Ctrl_seq[i,:])
            drone.piloting_pcmd(ctrl[0], ctrl[1], ctrl[2], ctrl[3], CtrlTime)
            time.sleep(CtrlTime)
            if takeoffsign and CAMERA_LIST:
                z = streaming_example.meta_other['ground_distance']
                centerstr = CAMERA_LIST[-1].split()
                camera_frame = find_xyz([int(centerstr[0]), int(centerstr[1])], z)
                pose_list.append(camera_frame)
            if cur_count == Checkpoint[cpind]:
                cmd_img_dir = os.path.join(Output_root, 'image', str(cur_count))
                cmd_dict_dir = os.path.join(Output_root, 'meta', str(cur_count))
                cmd_ctrl_dir = os.path.join(Output_root, 'ctrl', str(cur_count))
                create_dir([cmd_img_dir, cmd_dict_dir, cmd_ctrl_dir])
                meta = read_meta(streaming_example.meta_other, CAMERA_LIST[-1])
                img = streaming_example.current_frame
                data_diff = diff(meta, Metadata[cur_count])
                print('data_diff', data_diff)
                if np.linalg.norm(data_diff[0]) < xyz_norm and abs(data_diff[1]) < orient_difference_threshold:
                    print("No need corrections. Continue for the next checkpoint")
                    np.savetxt(os.path.join(cmd_ctrl_dir, str(corr_count)+'ctrl.txt'),np.array(ctrl_seq))
                    cpind += 1
                    continue
                print("Need corrections. Corrections begin")
                ctrl_seq = []
                # np.savetxt(os.path.join(cmd_ctrl_dir, str(corr_count)+'data_diff.txt'),np.array(data_diff), fmt='%1.4f', newline='\n')
                cv2.imwrite(os.path.join(cmd_img_dir, str(corr_count)+'error.png'), img)
                np.savetxt(os.path.join(cmd_dict_dir, str(corr_count)+'error.txt'),np.array(meta), fmt='%1.4f', newline='\n')
                takeoffsign = False
                while True:
                    if control.break_loop():
                        print("Finish correction")
                        cv2.imwrite(os.path.join(cmd_img_dir, str(corr_count)+'after_cor.png'), img)
                        np.savetxt(os.path.join(cmd_dict_dir, str(corr_count)+'after_cor.txt'),np.array(meta), fmt='%1.4f', newline='\n')
                        break
                    elif control.takeoff():
                        takeoffsign = True
                        drone(TakeOff(_no_expect=True) & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                    elif control.landing():
                        takeoffsign = False
                        drone(Landing() >> FlyingStateChanged(state="landed", _timeout=5)).wait()
                    elif control.has_piloting_cmd():   # correction
                        img = streaming_example.current_frame
                        meta = read_meta(streaming_example.meta_other, CAMERA_LIST.pop(-1).split())
                        data_diff = diff(meta, Metadata[cur_count])
                        print("The different is: ", data_diff)
                        drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), CtrlTime)
                        data_diff.extend([control.roll(), control.pitch(), control.yaw(), control.throttle()])
                        ctrl_seq.append(data_diff)
                        if np.linalg.norm(data_diff[0]) < xyz_norm and abs(data_diff[1]) < orient_difference_threshold:
                            print("Finish correction")
                            cv2.imwrite(os.path.join(cmd_img_dir, str(corr_count)+'after_cor.png'), img)
                            np.savetxt(os.path.join(cmd_dict_dir, str(corr_count)+'after_cor.txt'),np.array(meta), fmt='%1.4f', newline='\n')
                            np.savetxt(os.path.join(cmd_ctrl_dir, str(corr_count)+'ctrl.txt'),np.array(ctrl_seq), fmt='%1.4f %1.4f %1.4f %d %d %d %d', newline='\n')
                            break
                    if takeoffsign and CAMERA_LIST:
                        z = streaming_example.meta_other['ground_distance']
                        centerstr = CAMERA_LIST.pop(-1).split()
                        camera_frame = find_xyz([int(centerstr[0]), int(centerstr[1])], z)
                        pose_list.append(camera_frame)
                np.save('./train_data_comp_new/poselist.npy', np.array(pose_list))
                cpind += 1
        print("Landing for next round of correction")
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
