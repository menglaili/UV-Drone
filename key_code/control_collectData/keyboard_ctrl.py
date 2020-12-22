import subprocess
import time
from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import cv2
import numpy as np
# import matplotlib.pyplot as plt



class Ctrl(Enum):
    (
        QUIT,
        TAKEOFF,
        LANDING,
        MOVE_LEFT,
        MOVE_RIGHT,
        MOVE_FORWARD,
        MOVE_BACKWARD,
        MOVE_UP,
        MOVE_DOWN,
        TURN_LEFT,
        TURN_RIGHT,
        BREAK_LOOP,
        WAIT,
        CHECK,
    ) = range(14)


QWERTY_CTRL_KEYS = {
    Ctrl.QUIT: Key.esc,
    Ctrl.TAKEOFF: "t",
    Ctrl.LANDING: "l",
    Ctrl.MOVE_LEFT: "a",
    Ctrl.MOVE_RIGHT: "d",
    Ctrl.MOVE_FORWARD: "w",
    Ctrl.MOVE_BACKWARD: "s",
    Ctrl.MOVE_UP: Key.up,
    Ctrl.MOVE_DOWN: Key.down,
    Ctrl.TURN_LEFT: Key.left,
    Ctrl.TURN_RIGHT: Key.right,
    Ctrl.BREAK_LOOP: Key.enter,
    Ctrl.WAIT: Key.space,
    Ctrl.CHECK: "c",
}


KEY_POS = {
    Key.esc: [1620, 1860, 250, 380],
    Key.enter: [1710, 1860, 390, 670],
    "t": [680, 810, 390, 520],
    "l": [1220, 1340, 530, 660],
    "w": [310, 440, 390, 520],
    "a": [220, 350, 530, 660],
    "s": [350, 470, 530, 660],
    "d": [470, 590, 530, 660],
    Key.up: [2040, 2160, 670, 800],
    Key.down: [2040, 2160, 810, 940],
    Key.left: [1910, 2040, 810, 940],
    Key.right: [2160, 2290, 810, 940],
    Key.space: [570, 1296, 809, 943],
    None: False
}


class KeyboardCtrl(Listener):
    def __init__(self, ctrl_keys=None):
        self._ctrl_keys = self._get_ctrl_keys(ctrl_keys)
        self._key_pos = KEY_POS
        self._key_pressed = defaultdict(lambda: False)   # any new key would be given a False value
        self._last_action_ts = defaultdict(lambda: 0.0)
        self.keyimage = cv2.imread('key.jpg')
        # cv2.imshow('Keyboard', self.keyimage)
        self._keying = None
        super().__init__(on_press=self._on_press, on_release=self._on_release)
        self.start()

    def _on_press(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = True
            self._keying = key
            print(self._keying)
        elif isinstance(key, Key):
            self._key_pressed[key] = True
            self._keying = key
            print(self._keying)
        return True

    def _on_release(self, key):
        print(self._key_pressed[self._ctrl_keys[Ctrl.QUIT]])
        if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
            # self._key_pressed[self._ctrl_keys[Ctrl.QUIT]] = False
            return False
        else:
            if isinstance(key, KeyCode):
                self._key_pressed[key.char] = False
                self._keying = None
            elif isinstance(key, Key):
                self._key_pressed[key] = False
                self._keying = None
            return True


    def showkey(self):
        
        if not self._keying:
            print('cond1:',self._keying)
            cv2.imshow('Keyboard', self.keyimage)
        else:
            print('cond2:',self._keying)
            keyind = self._key_pos[self._keying]
            temp = np.copy(self.keyimage[keyind[2]:keyind[3], keyind[0]:keyind[1]])
            self.keyimage[keyind[2]:keyind[3], keyind[0]:keyind[1], 1:3] = np.array([0, 0])
            cv2.imshow('Keyboard', self.keyimage)
            self.keyimage[keyind[2]:keyind[3], keyind[0]:keyind[1]] = temp
        return True

    def checkpoint(self):
        # mark the checkpoint
        return self._key_pressed[self._ctrl_keys[Ctrl.CHECK]]

    def wait(self):
        return self._key_pressed[self._ctrl_keys[Ctrl.WAIT]]

    def quit(self):
        return self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]

    def _axis(self, left_key, right_key):  # control the speed of moving  [-100,100]
        return 30 * (
            int(self._key_pressed[right_key]) - int(self._key_pressed[left_key])
        )

    def roll(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_LEFT],
            self._ctrl_keys[Ctrl.MOVE_RIGHT]
        )

    def pitch(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_BACKWARD],
            self._ctrl_keys[Ctrl.MOVE_FORWARD]
        )

    def yaw(self):
        return 2 * self._axis(
            self._ctrl_keys[Ctrl.TURN_LEFT],
            self._ctrl_keys[Ctrl.TURN_RIGHT]
        )

    def throttle(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_DOWN],
            self._ctrl_keys[Ctrl.MOVE_UP]
        )

    def has_piloting_cmd(self):
        return (
            bool(self.roll())
            or bool(self.pitch())
            or bool(self.yaw())
            or bool(self.throttle())
        )

    def _rate_limit_cmd(self, ctrl, delay):
        now = time.time()
        if self._last_action_ts[ctrl] > (now - delay):
            return False
        elif self._key_pressed[self._ctrl_keys[ctrl]]:
            self._last_action_ts[ctrl] = now
            return True
        else:
            return False

    def takeoff(self):
        return self._rate_limit_cmd(Ctrl.TAKEOFF, 2.0)

    def landing(self):
        return self._rate_limit_cmd(Ctrl.LANDING, 2.0)

    def break_loop(self):
        return self._rate_limit_cmd(Ctrl.BREAK_LOOP, 1.0)

    def _get_ctrl_keys(self, ctrl_keys):
        # Get the default ctrl keys based on the current keyboard layout:
        if ctrl_keys is None:
            ctrl_keys = QWERTY_CTRL_KEYS
            
        return ctrl_keys


if __name__ == "__main__":
    import olympe
    from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
    ctrl_seq = []
    with olympe.Drone("192.168.42.1") as drone:
        drone.connection()
        drone.start_piloting()
        control = KeyboardCtrl()
        while not control.quit():
            if control.takeoff():
                drone(TakeOff())
            elif control.landing():
                drone(Landing())
            elif control.has_piloting_cmd() or control.wait():
                if control.has_piloting_cmd():
                    ctrl_seq.append([control.roll(), control.pitch(), control.yaw(), control.throttle(), 0])  # 0 stands for not hover
                    drone.piloting_pcmd(control.roll(), control.pitch(), control.yaw(), control.throttle(), 0.02)
                else:  # when fly over the area needs to be clean
                    ctrl_seq.append([control.roll(), control.pitch(), control.yaw(), control.throttle(), 1])  # 1 stands for hover
                    drone.piloting_pcmd(0, 0, 0, 0, 0.02)
            time.sleep(0.02)
