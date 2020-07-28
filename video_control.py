import csv
import cv2
import math
import os
import queue
import shlex
import subprocess
import threading
import traceback
import time
import numpy as np
import pickle
import pika




import olympe
from olympe.messages.ardrone3.Piloting import moveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
from olympe.enums.ardrone3.PilotingState import FlyingStateChanged_State
from olympe.messages.ardrone3.GPSSettingsState import HomeChanged
from olympe.messages.ardrone3.GPSSettingsState import GPSFixStateChanged
from olympe.messages.ardrone3.PilotingState import PositionChanged
from olympe.messages.camera import set_recording_mode, recording_mode

from keyboard_ctrl import KeyboardCtrl
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from pynput.keyboard import Listener, Key, KeyCode


olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})

def read_meta(metadata):
    w = metadata['drone_quat']['w']
    x = metadata['drone_quat']['x']
    y = metadata['drone_quat']['y']
    z = metadata['drone_quat']['z']
    g_d = metadata['ground_distance']
    exp_meta = [g_d,w,x,y,z]
    return exp_meta


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

class StreamingExample(threading.Thread):

    def __init__(self, drone, recording, path = [(1,0),(2, 90)]):
        # Create the olympe.Drone object from its IP address
        self.drone = drone  #10.202.0.1  192.168.42.1
        self.datafile = './data_new'
        
        if not os.path.isdir(self.datafile):
            os.mkdir(self.datafile)
        if not os.path.isdir(os.path.join(self.datafile,'image')):
            os.mkdir(os.path.join(self.datafile,'image'))
        if not os.path.isdir(os.path.join(self.datafile,'metadata')):
            os.mkdir(os.path.join(self.datafile,'metadata'))

        self.h264_frame_stats = []
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
        self.recording = recording
        self.path = path  # path file e.g. [(1,0),(2, 90)]
        self.flytimestamp = 0.02  # one command duration
        self.resolution = 0.75


        super().__init__()
        super().start()

    def dis2pair(self):
        new_pair = []
        for disall in self.path:
            dis = disall[0]
            deg = disall[1]
            if len(disall)>2:
                lat = disall[2]
            for i in range(1,10):
                if dis/i <= 1:
                    if len(disall)<=2:
                        new_pair.append((dis/i, math.radians(deg)))
                        for j in range(1,i):
                            new_pair.append((dis/i, 0))
                        break
                    else:
                        new_pair.append((dis/i, math.radians(deg), lat))
                        for j in range(1,i):
                            new_pair.append((dis/i, 0, lat))
                        break
        self.path = new_pair

    def start(self):
        # Connect the the drone
        self.drone.connect()
        # self.drone(GPSFixStateChanged(_policy = 'wait'))
        # print("GPS position before take-off :", self.drone.get_state(HomeChanged))

        # You can record the video stream from the drone if you plan to do some
        # post processing.
        if self.recording:
            if self.recording:
                self.drone.set_streaming_output_files(
                    h264_data_file=os.path.join(self.datafile, 'h264_data.264'),
                    # Here, we don't record the (huge) raw YUV video stream
                    # raw_data_file=os.path.join(self.datafile,'raw_data.bin'),
                    # raw_meta_file=os.path.join(self.datafile,'raw_metadata.json'),
                )
            else:
                self.drone.set_streaming_output_files()

            # Setup your callback functions to do some live video processing
            self.drone.set_streaming_callbacks(
                raw_cb=self.yuv_frame_cb,
                flush_raw_cb=self.flush_cb,
            )
            # Start video streaming
            self.drone.start_video_streaming()

    def stop(self):
        if self.recording:
            # Properly stop the video stream and disconnect
            self.drone.stop_video_streaming()
            self.drone.disconnect()
        else:
            pass

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.
            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def flush_cb(self):
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True

    # def get_frame(self):
    # 	return self.current_frame

    def show_yuv_frame(self, window_name, yuv_frame):
        # the VideoFrame.info() dictionary contains some useful information
        # such as the video resolution
        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]

        self.meta_other = {key: yuv_frame.vmeta()[1][key] for key in ['drone_quat', 'ground_distance']}
        # self.meta_other = yuv_frame.vmeta()[1]
        # yuv_frame.vmeta() returns a dictionary that contains additional
        # metadata from the drone (no GPS coordinates(banned), battery percentage, ...)

        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]

        # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
        # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

        # Use OpenCV to convert the yuv frame to RGB
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
        # Use OpenCV to show this frame
        if self.resolution != 1:
            self.current_frame = cv2.resize(cv2frame, (0,0), fx=self.resolution, fy =self.resolution)
        else:
            self.current_frame = cv2frame
        cv2.imshow(window_name, cv2frame)
        cv2.waitKey(1)  # please OpenCV for 1 ms...

    def run(self):
        window_name = "Olympe Streaming Example"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        main_thread = next(
            filter(lambda t: t.name == "MainThread", threading.enumerate())
        )
        while main_thread.is_alive():
            with self.flush_queue_lock:
                try:
                    yuv_frame = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                try:
                    self.show_yuv_frame(window_name, yuv_frame)
                except Exception:
                    # We have to continue popping frame from the queue even if
                    # we fail to show one frame
                    traceback.print_exc()
                finally:
                    # Don't forget to unref the yuv frame. We don't want to
                    # starve the video buffer pool
                    yuv_frame.unref()
        cv2.destroyWindow(window_name)

    # the one for expert
    def fly_ep(self):
        cmd_img_dir = os.path.join(self.datafile,'image')
        cmd_dict_dir = os.path.join(self.datafile,'metadata')
        ctrl_seq = []
        self.drone.start_piloting()
        control = KeyboardCtrl()
        count = 0
        count_pilotcmd = 0
        nowtime = 0
        checkpoint = []
        pose_list = []
        takeoffsign = False
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        while not control.quit():
            if control.takeoff():
                print(2)
                takeoffsign = True
                self.drone(TakeOff(_no_expect=True)
                    & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
            elif control.landing():
                takeoffsign = False
                self.drone(Landing())
            elif control.has_piloting_cmd():
                count_pilotcmd += 1
                ctrl_seq.append([count, control.roll(), control.pitch(), control.yaw(), control.throttle()])  # 0 stands for not hover
                self.drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), self.flytimestamp)
                count += 1
            elif control.checkpoint():  # the next count number corresponding to the next control command
                ctime = time.time()
                if (len(checkpoint) == 0 or checkpoint[-1] != count) and (ctime-nowtime)>1:  # 2 seconds cool down time
                    checkpoint.append(count)
                    nowtime = ctime
                    meta = read_meta(self.meta_other)
                    camera_frame = find_xyz([int(centerstr[0]), int(centerstr[1])], meta[0])
                    meta1 = np.hstack((camera_frame, np.array(meta)[1:]))  # x,y,z,qw,qx,qy,qz  z is from camera frame
                    cv2.imwrite(os.path.join(cmd_img_dir, str(count)+'.png'), self.current_frame)
                    np.savetxt(os.path.join(cmd_dict_dir, str(count)+'.txt'),np.array(meta1), fmt='%1.4f', newline='\n')
                    ctrl_seq.append([count, 0, 0, 0, 0])
                    count += 1
            else:
                self.drone.piloting_pcmd(0, 0, 0, 0, self.flytimestamp)
                if count_pilotcmd != 0:
                    ctrl_seq.append([count, 0, 0, 0, 0])
                    count += 1
            time.sleep(self.flytimestamp)
            # record the trajectory
            if takeoffsign and CAMERA_LIST:
                z = self.meta_other['ground_distance']
                # center = np.loadtxt('./xy_pos.txt')
                centerstr = CAMERA_LIST.pop(0).split()
                camera_frame = find_xyz([int(centerstr[0]), int(centerstr[1])], z)
                pose_list.append(np.hstack((count, camera_frame)))

        np.savetxt(os.path.join(self.datafile, 'ctrl_seq.txt'),np.array(ctrl_seq), fmt='%d', newline='\n')
        np.savetxt(os.path.join(self.datafile, 'checkpoint.txt'), np.array(checkpoint), fmt='%d')
        np.savetxt(os.path.join(self.datafile, 'pose_list.txt'), np.array(pose_list), fmt='%d %.4f %.4f %.4f')
    

    def hang(self):
        control = KeyboardCtrl()
        self.drone.start_piloting()
        count = 0
        while not control.quit():
            if control.takeoff():
                self.drone(TakeOff())
            elif control.landing():
                self.drone(Landing())
            elif control.has_piloting_cmd():
                self.drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), 0.02)
                time.sleep(0.02)
                # print("GPS position after take-off : ", self.drone.get_state(PositionChanged))
            elif control.break_loop():
                cv2.imwrite(os.path.join(self.datafile, str(count)+'.png'), self.current_frame)
                count += 1
            

    def postprocessing(self):
        if self.recording:
            # Convert the raw .264 file into an .mp4 file
            h264_filepath = os.path.join(self.datafile, 'h264_data.264')
            mp4_filepath = os.path.join(self.datafile, 'h264_data.mp4')
            subprocess.run(
                shlex.split('ffmpeg -i {} -c:v copy -y {}'.format(
                    h264_filepath, mp4_filepath)),
                check=True
            )
        else:
        	pass
        # Replay this MP4 video file using the default video viewer (VLC?)
        # subprocess.run(
        #     shlex.split('xdg-open {}'.format(mp4_filepath)),
        #     check=True
        # )




if __name__ == "__main__":
    path = [(3.8, 0),(1.5, -60), (0.7, 0), (5, -95),(2.14, -80)]  #,(0, 90, 1.2), (0, 90), (0, 90), (0, 90, -0.7), 
    # path = [(0.5, 0), (0.5, 180)]
    with olympe.Drone("192.168.42.1") as drone:
        streaming_example = StreamingExample(drone, True, path)
        streaming_example.dis2pair()
        # print(streaming_example.path)
        # Start the video stream
        streaming_example.start()
        # Perform some live video processing while the drone is flying
        streaming_example.fly_ep() # streaming_example.fly() # streaming_example.hang()# 
        # Stop the video stream
        streaming_example.stop()
        # Recorded video stream postprocessing
        streaming_example.postprocessing()



'''
# Load data (deserialize)
with open(os.path.join('./data/metadata', str(num)+'.pickle'), 'rb') as handle:
    data = pickle.load(handle)
'''

'''
# the one for demo
    def fly(self):
        
        print("flying")
        self.drone(
            TakeOff(_no_expect=True)
            & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
        control = KeyboardCtrl()
        while self.path:
            movestep = self.path.pop(0)
            if len(self.path) != 0:
                print("next move step:", self.path[0]) 
            else:
                print(None)
            self.drone(
                moveBy(0, 0, 0, movestep[1])
                >> FlyingStateChanged(state="hovering", _timeout=5)
            ).wait()
            self.drone(
                moveBy(movestep[0], 0, 0, 0)
                >> FlyingStateChanged(state="hovering", _timeout=5)
            ).wait()
            if len(movestep)>2:
                self.drone(
                    moveBy(0, 0, movestep[2], 0)
                    >> FlyingStateChanged(state="hovering", _timeout=5)
                ).wait()
            # self.drone.wait(10)
            print("Press key to continue: ")
            while len(self.path)>0:
                if ((self.drone.get_state(FlyingStateChanged)["state"] is not
            FlyingStateChanged_State.hovering)) and control.takeoff():
                    print("retaking off!!")
                    self.drone(TakeOff())
                if control.hovering():
                    print("hovering begin!!")
                    time.sleep(10)
                    print("hovering end!!")
                if control.break_loop():
                    print("break!!")
                    break
                if control.landing():
                    self.drone(Landing())
                if control.has_piloting_cmd():
                    self.drone(
                        PCMD(
                            1,
                            control.roll(),
                            control.pitch(),
                            control.yaw(),
                            control.throttle(),
                            timestampAndSeqNum=0,
                        )
                    )
                else:
                    self.drone(PCMD(0, 0, 0, 0, 0, timestampAndSeqNum=0))
                time.sleep(0.05)

        print("Landing...")
        self.drone(
            Landing()
            >> FlyingStateChanged(state="landed", _timeout=5)
        ).wait()
        print("Landed\n")
'''