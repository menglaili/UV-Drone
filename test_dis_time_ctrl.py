# test dis ctrl and time ctrl
# Goal:
# 1. can choose how many step to move and land 
# 2. calculate the distance from the webcam and sum them up for average
# 3. store the data and draw the error-bar plot

# Need to do:
# 0. set up an orientation for the output coordinate at the first start point: Done
# 1. test the webcam localization accuracy: Done 10cm
# 2. build up the communication channel to get the center point and the W value
# 3. control the drone to fly in some pattern: set the 5 magnitude, set the 4 direction, set 10 time loop
# 4. store distance (x,y) and find the difference in each axis data and draw error-bar plot
# store in 2 list for time and distance, each list contain 4 dictionary the key is the ground truth command, the value is the (dx, dy)


'''
moveby(go forward, go rightward, go downward)
'''
import time
import pika
import pickle
import numpy as np
from pyquaternion import Quaternion
from extrinsct_webcam import pixel2world
import threading

from multiprocessing import Process
from multiprocessing import Pool
from video_control import StreamingExample


import olympe
from video_control import StreamingExample
from keyboard_ctrl import KeyboardCtrl
from pynput.keyboard import Listener, Key, KeyCode
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.enums.ardrone3.PilotingState import FlyingStateChanged_State
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='hello')
CAMERA_LIST = []
class ConsumerThread(threading.Thread):
    def __init__(self, host, *args, **kwargs):
        super(ConsumerThread, self).__init__(*args, **kwargs)
        self._host = host
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

def get3D(z):
    centerstr = CAMERA_LIST.pop(-1).split()
    return pixel2world(z, np.array(centerstr))

# magnitude
dis_mag = [0.1,0.2,0.3,0.4,0.5]
time_mag = [1,3,5]
# loop times
LOOPs = 10


def gen_ctrl_command(direction, mag):
    # direction: 0,1,2,3 forward, backward, , leftward rightward
    ctrl_direction = np.zeros(2)  
    if direction == 0:
        ctrl_direction[0] = mag
    elif direction == 1:
        ctrl_direction[0] = -mag
    elif direction == 2:
        ctrl_direction[1] = -mag
    elif direction == 3:
        ctrl_direction[1] = mag
    return ctrl_direction

def read_orimeta(metadata):
    w = metadata['w']
    x = metadata['x']
    y = metadata['y']
    z = metadata['z']
    return np.array([w,x,y,z])

def ori_diff(orien_ori, orien_end):
    orien_ori = read_orimeta(orien_ori)
    orien_end = read_orimeta(orien_end)
    q_meta = Quaternion(array=np.array(orien_ori))
    q_exp = Quaternion(array=np.array(orien_end))
    q_diff = Quaternion.absolute_distance(q_exp, q_meta) 
    return q_diff

messthreads = ConsumerThread("localhost")
print("start multiprocessing")
messthreads.start()
def fly_dis(drone, direction, streaming_example, data_dic = {}):

    # take off
    print("flying")
    drone(
        TakeOff(_no_expect=True)
        & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
    drone.start_piloting()
    # record the start point x,y
    z = streaming_example.meta_other['ground_distance']
    startpoint = get3D(z)
    orien_ori = streaming_example.meta_other['drone_quat']
    # while loop and execute the rest command in the command list
    control = KeyboardCtrl()
    for i, mag in enumerate(dis_mag):
        LOOP = 10
        data_dic[mag] = []
        ctrl_direction = gen_ctrl_command(direction, mag)
        while not control.quit():
            if control.takeoff():
                drone(TakeOff(_no_expect=True)
                        & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                z = streaming_example.meta_other['ground_distance']
                startpoint = get3D(z) 
            elif control.landing():
                drone(Landing())
            elif control.has_piloting_cmd():
                orien_end = streaming_example.meta_other['drone_quat']
                print('ori_diff: ', ori_diff(orien_ori, orien_end))
                drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), 0.02)
            elif control.checkpoint(): # run the command
                print(ctrl_direction)
                drone(
                    moveBy(ctrl_direction[0], ctrl_direction[1], 0, 0)
                    >> FlyingStateChanged(state="hovering", _timeout=5)
                ).wait()
                time.sleep(1)
                # record the difference
                z = streaming_example.meta_other['ground_distance']
                endpoint = get3D(z)
                data_dic[mag].append((endpoint - startpoint)[:2])
                startpoint = endpoint
                LOOP -= 1
            if LOOP == 0:
                break
        print(data_dic[mag])
        with open("./intermediu/dic_dir%d_mag%.1f.pickle" % (direction, mag), "wb") as f:
            pickle.dump(data_dic[mag], f)


if __name__ == "__main__":
    # store data
    dis_list = [{},{},{},{}]
    time_list = [{},{},{},{}]
    with olympe.Drone("192.168.42.1") as drone:
        streaming_example = StreamingExample(drone, True)
        streaming_example.start()
        for i in range(4):
            fly_dis(drone, i, streaming_example, data_dic = dis_list[i])
            with open("./dis_list_dic%d.pickle"%(i), "wb") as f:
                pickle.dump(dis_list, f)
        streaming_example.stop()