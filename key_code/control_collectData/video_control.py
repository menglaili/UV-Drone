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
import copy
import pika




import olympe
from olympe.messages.ardrone3.Piloting import moveBy, moveTo
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
from extrinsct_webcam import pixel2world
from pyquaternion import Quaternion
from NN_pred import *



olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})


# Receiving localization from Webcam
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

def read_meta(metadata):
    w = metadata['drone_quat']['w']
    x = metadata['drone_quat']['x']
    y = metadata['drone_quat']['y']
    z = metadata['drone_quat']['z']
    g_d = metadata['ground_distance']
    exp_meta = [g_d,w,x,y,z]
    return exp_meta

def read_orimeta(metadata):
    w = metadata['w']
    x = metadata['x']
    y = metadata['y']
    z = metadata['z']
    return np.array([w,x,y,z])

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


class StreamingExample(threading.Thread):
    '''
    changing to new scene: yaw0, path, data_dir, new tran_mats and explicit_para.py, ref_dir
    '''

    def __init__(self, drone, recording, path = [(1,0),(2, 90)]):
        # Create the olympe.Drone object from its IP address
        self.drone = drone  #10.202.0.1  192.168.42.1
        self.datafile = './data_new'
        
        if not os.path.isdir(self.datafile):
            os.mkdir(self.datafile)

        self.h264_frame_stats = []
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
        self.recording = recording
        self.path = path  # path file e.g. [(1,0),(2, 90)]
        self.flytimestamp = 0.02  # one command duration
        self.resolution = 0.75
        self.pathdic = {}
        self.yaw0 = 2.16# for the meeting room np.radians(64.903)
        self.ref_dir = "./test_scene1_ref"
        self.refinfo_stored = False
        self.SetSaveFlag()


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
        if self.recording:
            if self.recording:
                self.drone.set_streaming_output_files(
                    h264_data_file=os.path.join(self.datafile, 'h264_data.264'),
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
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def flush_cb(self):
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True


    def show_yuv_frame(self, window_name, yuv_frame):

        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]

        self.meta_other = {key: yuv_frame.vmeta()[1][key] for key in ['drone_quat', 'ground_distance']}

        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]


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
                    traceback.print_exc()
                finally:
                    yuv_frame.unref()
        cv2.destroyWindow(window_name)

    def hang(self):
        control = KeyboardCtrl()
        self.drone.start_piloting()
        count = 0
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        while not control.quit():
            if control.takeoff():
                self.drone(TakeOff())
            elif control.landing():
                self.drone(Landing())
            elif control.has_piloting_cmd():
                print(self.get_xyzq()[-1]) #  - self.yaw0
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

    def get_pose(self):
        z = self.meta_other['ground_distance']
        xyz = get3D(z)
        orien_ori = read_orimeta(self.meta_other['drone_quat'])
        return np.hstack((xyz, orien_ori))

    def get_xyzq(self):
        pose = self.get_pose()
        xyzq = self.pose2xyzq(pose)
        return xyzq

    def SaveImgPose(self):
        if self.SaveImgFlag:
            cv2.imwrite(self.pathdic["img"], self.current_frame)
        if self.SavePoseFlag:
            cur_pose = self.get_pose()
            with open(self.pathdic["pose"], "ab") as f:
                np.savetxt(f, cur_pose.reshape(1, -1), fmt = "%1.4f")

    def SavePose(self):
        if self.SavePoseFlag:
            cur_pose = self.get_pose()
            with open(self.pathdic["pose"], "ab") as f:
                np.savetxt(f, cur_pose.reshape(1, -1), fmt = "%1.4f")

    def SaveCtrl(self, action, count):
        if self.SaveCtrlFlag:
            if action is not None:
                with open(self.pathdic["ctrl"], "ab") as f:
                    if len(np.array(action).shape) == 1:
                        np.savetxt(f, np.hstack((count, np.array(action))).reshape(1, -1), fmt = "%1.4f")
                    else:
                        np.savetxt(f, np.hstack((count * np.ones((np.array(action).shape[0], 1)), np.array(action))), fmt = "%1.4f")

            else:
                with open(self.pathdic["ctrl"], "ab") as f:
                    f.write(b"None\n")
    
    def SaveDict(self, Dict, label):
        if self.SaveDictFlag:
            with open(self.pathdic["dict"][label], "wb") as f:
                pickle.dump(Dict, f)

    def SetSaveFlag(self, Flag = [1, 1, 1, 1]):
        self.SaveImgFlag = False
        self.SavePoseFlag = False
        self.SaveCtrlFlag = False
        self.SaveDictFlag = False
        if Flag[0]:
            self.SaveImgFlag = True
        if Flag[1]:
            self.SavePoseFlag = True
        if Flag[2]:
            self.SaveCtrlFlag = True
        if Flag[3]:
            self.SaveDictFlag = True

    def create_file(self, dir1, dir2 = None, create = False):
        if dir2:
            directory = os.path.join(dir1, dir2)
            if create:
                if not os.path.isdir(dir1):
                    os.mkdir(dir1)
                if not os.path.isdir(directory):
                    os.mkdir(directory)
            return directory
        else:
            if not os.path.isdir(dir1):
                os.mkdir(dir1)

    def SetParaDir(self, corrtimes, file_dir):
        # set up data dir
        # output and store setting parameter
        self.NNtestdir = self.create_file(file_dir, str(corrtimes))
        self.pathdic["dict"] = {}
        self.pathdic["dict"]["cpose"] = self.create_file(self.NNtestdir,'checkpoint_poses.pickle')
        self.pathdic["dict"]["ctrl_pred"] = self.create_file(self.NNtestdir,'ctrl_pred.pickle')
        self.pathdic["img"] = self.create_file(self.NNtestdir,'image', create = True)
        self.pathdic["pose"] = self.create_file(self.NNtestdir,"pose_list.txt")
        self.pathdic["ctrl"] = self.create_file(self.NNtestdir,"ref_ctrl_current.txt")

    def path2world(self):
        cur_xy = np.array((0.0, 0.0))
        turning = 1
        radians = 0
        height = 1
        self.track_list = [np.hstack((np.copy(cur_xy), 1, radians))]
        for _, data in enumerate(self.path):
            if len(data) == 2:
                forward, turn, dh = data[0], data[1], 0   
            elif len(data) == 3:
                forward, turn, dh = data[0], data[1], data[2]
            if turn != 0:
                turning = 1 - turning
            if turning:
                cur_xy[0] += int(np.cos(radians)) * forward
            else:
                cur_xy[1] += -int(np.sin(radians)) * forward
            radians += turn
            height += dh
            self.track_list.append(np.hstack((np.copy(cur_xy), height, radians)))
        self.track_list = np.array(self.track_list)
        # self.track_list[:, 3] += self.yaw0

    def pose2xyzq(self, pose):
        # print("yaw0, ", Quaternion(pose[3:]).yaw_pitch_roll[0])
        yaw = Quaternion(pose[3:]).yaw_pitch_roll[0] - self.yaw0
        return np.hstack((pose[:3], yaw))

    def compute_diff(self, count, onlyorihei = False):
        if CAMERA_LIST:
            cur_pose = self.get_pose()
            cur_xyzq = self.pose2xyzq(cur_pose)
            gt_xyzq = self.track_list[count, :]
            print("xyzq diff ", cur_xyzq - gt_xyzq)
            sign = False
            if not onlyorihei:
                if np.all(np.abs(cur_xyzq - gt_xyzq) < np.array((0.07, 0.07, 0.07, 0.03))):
                    sign = True
                return sign, cur_xyzq - gt_xyzq, cur_xyzq, gt_xyzq
            else:
                if np.all(np.abs(cur_xyzq - gt_xyzq)[2:] < np.array((0.1, 0.1, 0.1, 0.03))[2:]):
                    sign = True
                return sign
        else:
            if not onlyorihei:
                return False, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
            else:
                return False


    def check_refinfo(self):
        if not self.refinfo_stored:
            self.ref_data = {}
            self.ref_data["ref_pose"] = np.loadtxt(os.path.join(self.ref_dir, "pose.txt"))
            self.ref_data["ref_ctrl"] = np.loadtxt(os.path.join(self.ref_dir, "ref_ctrl.txt"))
            img_num = len(os.listdir(os.path.join(self.ref_dir, "image")))
            self.ref_data["ref_img"] = {}
            for i in range(img_num):
                self.ref_data["ref_img"][str(i)] = cv2.imread(os.path.join(self.ref_dir, "image", str(i) + ".png"))
            self.refinfo_stored = True

    def updateTL(self):
        # update track_list
        self.check_refinfo()
        self.track_list = []
        for i in range(self.ref_data["ref_pose"].shape[0]):
            self.track_list.append(self.pose2xyzq(self.ref_data["ref_pose"][i, :]))
        self.track_list = np.array(self.track_list)

    def xyzq2Tinv(self, xyzq, onlyR = False):
        q = -xyzq[-1]  # to change the angle to counter-clockwise
        R = np.array(([np.cos(q), -np.sin(q), 0],[np.sin(q), np.cos(q), 0],[0, 0, 1]))
        if onlyR:
            return R.T  
        else:
            Tinv = np.zeros((4, 4))
            Tinv[:3, :3] = R.T
            Tinv[:3, 3] = -R.T.dot(xyzq[:3])
            Tinv[3, 3] = 1
            return Tinv

    def tohomo(self, vec):
        return np.hstack((vec, 1))

    def pose2corr(self, count):
        gt_xyzq = self.track_list[count, :]
        gt_xyz = self.tohomo(gt_xyzq[:3])
        cur_pose = self.get_pose()
        cur_xyzq = self.pose2xyzq(cur_pose)
        cur_Tinv = self.xyzq2Tinv(cur_xyzq)
        gt_xyz_cur = cur_Tinv.dot(gt_xyz)[:3]
        gt_xyz_cur[1] = -gt_xyz_cur[1]  # rightward to leftward
        gt_xyz_cur[2] = -gt_xyz_cur[2]  # downward to upward
        q_diff = gt_xyzq[-1] - cur_xyzq[-1] 
        return np.hstack((gt_xyz_cur, q_diff)) # combine

    def t_orihei2corr(self, count):
        corr_orihei = self.pose2corr(count)[2:]
        return corr_orihei


    def NNrel2corr(self, count):
        # NN output is the relative pose with no rotation from current to reference
        self.check_refinfo()
        abs_trans = NNtrans(self.current_frame, self.ref_data["ref_img"][str(count)])  # return should be numpy array [deltax, deltay, deltaz]
        cur_xyzq = self.get_xyzq()
        Rinv = self.xyzq2Tinv(cur_xyzq, onlyR = True)
        cur_trans = Rinv.dot(abs_trans)
        corr_orihei = self.t_orihei2corr(count)
        cur_trans[1] = -cur_trans[1] # rightward to leftward
        z_pred = cur_trans[-1]
        return np.hstack((cur_trans[:2], corr_orihei)), z_pred, abs_trans


    def NNrel2corr1(self, count):
        # NN output is the relative pose with no rotation from current to reference
        self.check_refinfo()
        abs_trans = NNtrans1(self.current_frame, self.ref_data["ref_img"][str(count)], count)  # return should be numpy array [deltax, deltay, deltaz]
        abs_trans[0] = abs_trans[0]
        abs_trans[1] = abs_trans[1]
        # cur_xyzq = self.get_xyzq()
        # Rinv = self.xyzq2Tinv(cur_xyzq, onlyR = True)
        # cur_trans = Rinv.dot(self.tohomo(abs_trans))
        corr_orihei = self.t_orihei2corr(count)
        # z_pred = cur_trans[-1]

        return np.hstack((abs_trans, corr_orihei))


    def movebyseq(self, correction, onlyorihei = False):
        if onlyorihei:
            self.drone(moveBy(0, 0, 0, correction[1])
            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
            time.sleep(0.5)
            self.drone(moveBy(0, 0, correction[0], 0)
            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
            time.sleep(0.5)
        else:
            self.drone(moveBy(0, 0, 0, correction[3])
            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
            time.sleep(0.5)
            self.drone(moveBy(0, 0, correction[2], 0)
            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
            time.sleep(0.5)
            self.drone(moveBy(0, correction[1], 0, 0)
            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
            time.sleep(0.5)
            self.drone(moveBy(correction[0], 0, 0, 0)
            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
            time.sleep(0.5)


    def random_pilot(self, mark_rand, count):
        while True:
            rndval = np.random.uniform(0.15, 0.3)
            if mark_rand < 3:
                err_ctrl = np.array([rndval, 0, 0, 0])
                ind = 0
            elif mark_rand < 6:
                err_ctrl = np.array([-rndval, 0, 0, 0])
                ind = 0
            elif mark_rand < 9:
                err_ctrl = np.array([0, rndval, 0, 0])
                ind = 1
            elif mark_rand < 12:
                err_ctrl = np.array([0, -rndval, 0, 0])
                ind = 1
            gt_xyzq = self.track_list[count, :]
            cur_xyzq = self.get_xyzq()
            rndsign = False
            if 0.12 < abs((gt_xyzq-cur_xyzq)[:2][ind] + err_ctrl[ind]) < 0.35:
                rndsign = True
                print("total error: ", abs((gt_xyzq-cur_xyzq)[:2][ind] + err_ctrl[ind]))
                return err_ctrl, rndsign


    # experiment for open loop, apply several times distance control
    def fly_dis_traj(self):
        #OriAndHei (reference)qw, qx, qy, qz, (current)qw, qx, qy, qz, height, height, action_ori, action_height
        #Translation img_addr1, img_addr2, reltranx, reltrany, reltranz, gt_pose1, gt_pose2, action
        #Ref_ctrl: distance ctrl, Imgs, Poses
        CorrTime = 3
        control = KeyboardCtrl()
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        TemPath = copy.deepcopy(self.path)
        assert not os.path.isfile("./data_new/0/pose.txt"), "Exist pose.txt file"
        for i in range(CorrTime):
            self.datafile = "./data_new/" + str(i)
            cmd_img_dir = os.path.join(self.datafile,'image')
            cmd_dict_dir = os.path.join(self.datafile,'pose')
            cmd_ctrl_dir = os.path.join(self.datafile,'ref_ctrl')
            if not os.path.isdir(self.datafile):
                os.mkdir(self.datafile)
            if not os.path.isdir(cmd_img_dir):
                os.mkdir(cmd_img_dir)
            count = 0
            takeoffsign = False
            while (not control.quit()) and self.path:
                if control.takeoff():
                    takeoffsign = True
                    self.drone(TakeOff(_no_expect=True)
                        & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                    time.sleep(1)
                elif control.landing():
                    takeoffsign = False
                    self.drone(Landing())
                if takeoffsign:
                    self.pathdic["img"] = os.path.join(cmd_img_dir, str(count)+'.png')
                    self.pathdic["pose"] = cmd_dict_dir + '.txt'
                    self.pathdic["ctrl"] = cmd_ctrl_dir + '.txt'
                    action = self.path.pop(0)
                    print("Next action is ", action)
                    self.SaveImgPose()
                    self.SaveCtrl(action, count)
                    self.drone(moveBy(action[0], 0, 0, action[1])
                    >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
                    time.sleep(1)
                    count += 1
            self.pathdic["img"] = os.path.join(cmd_img_dir, str(count)+'.png')
            self.SaveImgPose() # save the last checkpoint
            self.path = copy.deepcopy(TemPath)
            self.drone(Landing())
    


    # collect reference data
    def get_dis_ref(self, file_dir = './reference_data'):
        '''
        the print orientation is already subtract yaw0, the store orientation does not
        '''
        self.datafile = file_dir
        if not os.path.isdir(self.datafile):
            os.mkdir(self.datafile)
        self.drone.start_piloting()
        control = KeyboardCtrl()
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        cmd_img_dir = os.path.join(self.datafile,'image')
        cmd_dict_dir = os.path.join(self.datafile,'pose')
        cmd_ctrl_dir = os.path.join(self.datafile,'ref_ctrl')
        if not os.path.isdir(cmd_img_dir):
            os.mkdir(cmd_img_dir)
        count = 0
        nowtime = 0
        self.path2world()
        meet_threshold = False
        self.path.append("End")
        if not CAMERA_LIST:
            self.SetSaveFlag([1, 0, 1, 0])
        while (not control.quit()) and self.path:
            if control.takeoff():
                self.drone(TakeOff(_no_expect=True)
                    & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                time.sleep(1)
            elif control.landing():
                self.drone(Landing())
            elif control.has_piloting_cmd():
                meet_threshold, _, _, _ = self.compute_diff(count)
                self.drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), 0.02)
                time.sleep(0.02)
            elif control.checkpoint() or meet_threshold:  # the next count number corresponding to the next control command
                ctime = time.time()
                if (ctime-nowtime)>1:  # 2 seconds cool down time
                    nowtime = ctime
                    self.pathdic["img"] = os.path.join(cmd_img_dir, str(count)+'.png')
                    self.pathdic["pose"] = cmd_dict_dir + '.txt'
                    self.pathdic["ctrl"] = cmd_ctrl_dir + '.txt'
                    action = self.path.pop(0)
                    print("Next action is ", action)
                    self.SaveImgPose()
                    if action != "End":
                        self.SaveCtrl(action, count)
                    count += 1
                    meet_threshold = False
        self.drone(Landing())


    # collect the training data without DAgger
    def get_dis_cor(self, file_dir = './initial_training_data'):
        Start_ind = 1
        randnum = 30
        CorrTime = Start_ind + randnum
        # self.randctrllist = np.random.uniform(0.1, 0.3, randnum)
        # assert not os.path.isdir("./data_new/%d" % Start_ind), "Exist data file"
        control = KeyboardCtrl()
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        TemPath = copy.deepcopy(self.path)
        self.updateTL()
        self.SetSaveFlag()
        for i in range(Start_ind, CorrTime):
            self.path.append("End")
            self.datafile = file_dir 
            if not os.path.isdir(self.datafile):
                os.mkdir(self.datafile)
            self.datafile += str(i)
            cmd_img_dir = os.path.join(self.datafile,'image')
            cmd_dict_dir = os.path.join(self.datafile,'pose')
            cmd_ctrl_dir = os.path.join(self.datafile,'ref_ctrl')
            if not os.path.isdir(self.datafile):
                os.mkdir(self.datafile)
            if not os.path.isdir(cmd_img_dir):
                os.mkdir(cmd_img_dir)
            if not os.path.isdir(cmd_dict_dir):
                os.mkdir(cmd_dict_dir)
            count = 0
            takeoffsign = False

            while self.path:
                if control.takeoff():
                    takeoffsign = True
                    self.drone(TakeOff(_no_expect=True)
                        & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                    time.sleep(1)
                elif control.landing():
                    takeoffsign = False
                    self.drone(Landing())
                elif control.break_loop():
                    meet_threshold = False
                    break
                if takeoffsign:
                    # mark_rand = (i - Start_ind)
                    # err_ctrl, rndsign = self.random_pilot(mark_rand, count)
                    # if rndsign:
                    #     self.movebyseq(err_ctrl)
                    meet_threshold, _, _, _ = self.compute_diff(count)  
                    while not meet_threshold:
                        corr_cur = self.pose2corr(count)[2:]
                        self.movebyseq(corr_cur, onlyorihei = True)
                        meet_threshold = self.compute_diff(count, onlyorihei = True)
                    print("orientation corrected")
                    self.pathdic["img"] = os.path.join(cmd_img_dir, str(count)+'current.png')
                    self.pathdic["pose"] = cmd_dict_dir + '_current.txt'
                    self.pathdic["ctrl"] = cmd_ctrl_dir + '_current.txt'
                    self.SaveImgPose()
                    meet_threshold, _, _, _ = self.compute_diff(count)
                    correction = None
                    if not meet_threshold:
                        correction = []
                        print("correction begins")
                        corr_count = 0
                        while True:
                            if control.takeoff():
                                takeoffsign = True
                                self.drone(TakeOff(_no_expect=True)
                                    & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                                time.sleep(1)
                            elif control.landing():
                                takeoffsign = False
                                self.drone(Landing())
                            if takeoffsign:
                                meet_threshold = self.compute_diff(count, onlyorihei = True)
                                while not meet_threshold:
                                    corr_cur = self.pose2corr(count)[2:]
                                    self.movebyseq(corr_cur, onlyorihei = True)
                                    meet_threshold = self.compute_diff(count, onlyorihei = True)
                                print("orientation corrected")
                                corr_cur = self.pose2corr(count)
                                # corr_cur = input()
                                correction.append(corr_cur)
                                print(corr_cur)
                                self.movebyseq(corr_cur)
                                self.pathdic["img"] = os.path.join(cmd_img_dir, str(count) + "_" + str(corr_count) + "_" + 'current.png')
                                self.pathdic["pose"] = os.path.join(cmd_dict_dir, str(count) + str(corr_count) +'current.txt')
                                self.SaveImgPose()
                                corr_count += 1
                                meet_threshold, _, _, _ = self.compute_diff(count)
                                if meet_threshold:
                                    meet_threshold = False
                                    break
                    print("meet requirement")
                    self.SaveCtrl(correction, count)
                    action = self.path.pop(0)
                    if action != "End":
                        print("This action is ", action)
                        self.drone(moveBy(action[0], 0, 0, action[1])
                        >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
                        time.sleep(1)
                    count += 1
            self.path = copy.deepcopy(TemPath)
            self.drone(Landing())

    

    # collect data with DAgger
    def test_NN_perform(self, file_dir = "./augmented_training_data"):
        correction_times_start = 137  # new model starting from 69  # scene1 starting from 135
        correction_times_end = 138
        control = KeyboardCtrl()
        messthreads = ConsumerThread("localhost")
        messthreads.start()
        TemPath = copy.deepcopy(self.path)
        self.updateTL()
        self.SetSaveFlag()
        self.create_file(file_dir)
        self.drone.start_piloting()
        for count in range(correction_times_start, correction_times_end):
            self.SetParaDir(count, file_dir)
            self.path.append("End")
            CorrPoseDict = {}
            CorrCtrlDict = {}
            takeoffsign = False
            count = 0
            while (not control.quit()) and self.path:
                if control.takeoff():
                    takeoffsign = True
                    self.drone(TakeOff(_no_expect=True)
                        & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                    time.sleep(1)
                elif control.landing():
                    takeoffsign = False
                    self.drone(Landing())
                elif control.break_loop():
                    meet_threshold = False
                    break
                if takeoffsign:
                    meet_threshold, err_xyzq, cur_xyzq, _ = self.compute_diff(count)  # change to the newly collected reference
                    CorrCtrlDict[str(count)] = []
                    CorrPoseDict[str(count)] = [cur_xyzq, err_xyzq] # pose before corr, pose before error, pose after corr, pose after corr error
                    cur_xyzq_after, err_xyzq_after = cur_xyzq, err_xyzq
                    correction = None
                    self.SavePose()
                    if not meet_threshold:
                        self.pathdic["img"] = self.create_file(self.NNtestdir, 'image/' + str(count) + 'current.png')
                        self.SaveImgPose()
                        print("correction begins")
                        corr_count = 0
                        while not(meet_threshold):  #and (corr_count <= CorrTimes): #  
                            if control.takeoff():
                                takeoffsign = True
                                self.drone(TakeOff(_no_expect=True)
                                    & FlyingStateChanged(state="hovering", _policy="wait", _timeout=5)).wait()
                                time.sleep(1)
                            elif control.landing():
                                takeoffsign = False
                                self.drone(Landing())
                            elif control.has_piloting_cmd():
                                self.drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), 0.02)
                                time.sleep(0.02)
                            if takeoffsign:
                                meet_threshold = self.compute_diff(count, onlyorihei = True)
                                while not meet_threshold:
                                    corr_cur = self.pose2corr(count)[2:]
                                    self.movebyseq(corr_cur, onlyorihei = True)
                                    meet_threshold = self.compute_diff(count, onlyorihei = True)
                                print("orientation corrected")
                                # meet_threshold, _, _, _ = self.compute_diff(count)
                                # while not meet_threshold:
                                #     corr_cur = self.pose2corr(count)
                                #     self.movebyseq(corr_cur)
                                #     meet_threshold, _, _, _ = selft.compute_diff(count)
                                corr_cur = self.NNrel2corr1(count)
                                gt_corr = self.pose2corr(count)
                                correction = None
                                print("predicted_correction: ", corr_cur)
                                print("GT_correction: ", gt_corr)
                                if np.any(np.abs(gt_corr[:2] - corr_cur[:2]) > 0.02):
                                    correction = gt_corr
                                    self.pathdic["img"] = self.create_file(self.NNtestdir, 'image/' + str(count) + "_" + str(corr_count) + "_" + 'current.png')
                                    self.SaveImgPose()
                                self.movebyseq(corr_cur)
                                CorrCtrlDict[str(count)].append(corr_cur)
                                meet_threshold, err_xyzq_after, cur_xyzq_after, _ = self.compute_diff(count)
                                corr_count += 1
                                if corr_count > 0:
                                    break
                    self.SaveCtrl(correction, count)      
                    CorrPoseDict[str(count)].extend([cur_xyzq_after, err_xyzq_after])
                    print("Correction finish")
                    self.SaveDict(CorrPoseDict, "cpose")
                    self.SaveDict(CorrCtrlDict, "ctrl_pred")
                    action = self.path.pop(0)
                    if action != "End":
                        print("This action is ", action)
                        if len(action) == 2:
                            self.drone(moveBy(action[0], 0, 0, action[1])
                            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
                        elif len(action) == 3:
                            self.drone(moveBy(action[0], 0, -action[2], action[1])
                            >> FlyingStateChanged(state="hovering", _timeout=5)).wait()
                        time.sleep(1)
                    count += 1
            self.path = copy.deepcopy(TemPath)
            self.drone(Landing())



if __name__ == "__main__":
    path = [(0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
        (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
        (0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
        (0.5, 0), (0.5, 0), (0, np.radians(-90))]
    with olympe.Drone("192.168.42.1") as drone: # "192.168.42.1" "10.202.0.1"
        streaming_example = StreamingExample(drone, True, path)
        # streaming_example.dis2pair()
        print(streaming_example.path)
        streaming_example.start()
        streaming_example.get_dis_ref() 
        streaming_example.stop()
        streaming_example.postprocessing()
