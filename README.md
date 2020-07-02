# Keyboard Control and Data Collection
**Authors:** Menglai Li

**01 July 2020**
# 1. Prerequisites
Since the Parrot Olympe SDK only supports Linux system, all the code is tested in Ubuntu 18.04. It should be compatiable for other Linux distributions.

## Olympe
Installation and tutorials can be find [here](https://developer.parrot.com/docs/olympe/userguide.html)

Please pay attention that the olympe is running in a virtualenv and all the package installed at this environment may not be used outside of it. 

## Sphinx
Installation and usages can be find [here](https://developer.parrot.com/docs/sphinx/installation.html). This is the parrot drone official simulation environment. **For the safty purpose please run the code at least once in Sphinx before deploy it to the physical one.** 

If for any reason you cannot connect to the Sphinx engine please see [Trouble shooting](https://developer.parrot.com/docs/sphinx/troubleshooting.html).

## Other requirements
Run the following code to install all dependent packages:

```
pip3 install pynput numpy pyquaternion opencv-python
```

# 2. Usage
## Connection
To switch between physical drone and simulated drone, please change the IP address in the bottom part of the code file. 

For the physical one(**connect to the drone's WiFi**): 
```
if __name__ == "__main__":
    with olympe.Drone("192.168.42.1") as drone:
```

For the simulated one(**already start the Sphinx**):
```
if __name__ == "__main__":
    with olympe.Drone(""10.202.0.1"") as drone:
```
## Control the drone by keyboard
Run the following command in the Olympe's virtualenv. 

```
python3 keyboard_ctrl.py
```


**Controlling commands:**
- `w`: Go forward; `s`: Go backward; `a`: Go leftward; `d`: Go rightward
- ↑: Go up; ↓: Go down; ←: Turn left; →: Turn right
- `t`: Take off; `l`: Landing; `q`: Quit

It's a time control manner same as using joystick. The responding time for each command is set to 0.02s i.e. the command will execute for 0.02s then check. The roll angle, pitch angle, yaw rotation speed, and throttle when moving are set to 30% of their maximum values.


## Record the reference path and observations
Run the following command in the Olympe's virtualenv. 

```
python3 video_control.py
```

When the screen shows the FPV camera frame on a new window and the command line prints *Ready for record the reference path*, it ready to start.

- Press `t` to take off
- Give some piloting commands e.g. `w``w``a`
- Press *space* to select the first checkpoint
- Give some piloting commands
- Press *space* to select the second checkpoint
- ...
- Press `l` to land
- Press `q` to quit the program

The checkpoints are where the drone's observations are stored. When correction, the drone will follow your control command and wait for your correction commands at each checkpoints.

Please select the checkpoints in step size of 0.3-0.5m, otherwise the correction maybe huge. Also do not select more than 10 checkpoint since when correction the battery may not be enough. The correction command should be one of the 8 piloting command at one time and the duration can depend.

The data will stored at `./data/image` for image data, `./data/metadata` for onboard sensors, `./data/ctrl_seq.txt` for reference control commands, and `./data/checkpoint.txt` for checkpoint index.

**Emergency**
- when wanting the drone to land immediately, please press `Ctrl-C` to shut down the program and run `python3 keyboard_ctrl.py` then press `l`.

## Apply corrections to the reference path
Run the following command in the Olympe's virtualenv. 

```
python3 follow_ctrl.py
```

When the screen shows the FPV camera frame on a new window and the command line prints *Ready to Take off*, it ready to start.

- "Ready to Take off"
- Press `t` to take off
- "The next control command is: [30, 0, 0, 0]"        # the drone follows this command and fly
- "The different of observation at this checkpoint before correction: [0.6634, 0.003, 0.015]"     # the drone flys at the checkpoint, shows the difference
- "Need corrections. Corrections begin"  # if "No need corrections. Continue for the next checkpoint" then no corrections are needed
- Give some correction piloting commands
- "The difference of observation is: [0.67, 0.002, 0.016]"
- Give some correction piloting commands
- "The different of observation is: [0.68, 0.001, 0.013]"
- ...
- "Finish correction" or if you cannot reach the threshold anyway, press *Enter* to finish the correction
- "Landing for next round of correction" if finish all the checkpoint at this round
- One more round ...

The observation difference is **[Image matching ratio, orientation error, height error]**. The threshold is set to be **[0.70, 0.005, 0.015]**.

The next control command has the format **[roll, pitch, yaw, throttle]**. 

**Emergency**
- when wanting the drone to land immediately, please press `Ctrl-C` to shut down the program and run `python3 keyboard_ctrl.py` then press `l`.
