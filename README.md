# Keyboard Control and Data Collection
**Authors:** Menglai Li

**01 July 2020**
# 1. Prerequisites
Since the Parrot Olympe SDK only supports Linux system, all the code is tested in Ubuntu 18.04. It should be compatiable for other Linux distributions.

## Olympe
Installation and tutorials can be find [here](https://developer.parrot.com/docs/olympe/userguide.html)

Please pay attention that the olympe is running in a virtualenv and all the package installed at this environment may not be used outside of it. 

## Other requirements
Run the following code to install all dependent packages:

`
pip3 install -r requirements.txt --no-index --find-links file:///tmp/packages
`
## Sphinx
Installation and usages can be find [here](https://developer.parrot.com/docs/sphinx/installation.html). This is the parrot drone official simulation environment. For the safty purpose please run the code at least once at this environment before deploy it to the physical one. 

If for any reason you cannot connect to the Sphinx engine please see [Trouble shooting](https://developer.parrot.com/docs/sphinx/troubleshooting.html).

# 2. Usage
## Connection
To switch between physical drone and simulated drone, please change the IP address in the bottom part of the code file.

For the physical one:
```
if __name__ == "__main__":
    with olympe.Drone("192.168.42.1") as drone:
````

For the simulated one:
```
if __name__ == "__main__":
    with olympe.Drone(""10.202.0.1"") as drone:
```
## Control the drone by keyboard
Run the following command at the Olympe's virtualenv. 

`
python3 keyboard_ctrl.py
`


**Controlling commands:**
- `w`: Go forward; `s`: Go backward; `a`: Go leftward; `d`: Go rightward
- ↑: Go up; ↓: Go down; ←: Turn left; →: Turn right
- `t`: Take off; `l`: Landing; `q`: Quit

It's a time control manner same as using joystick. The responding time for each command is set to 0.02s i.e. the command will execute for 0.02s then check. The roll angle, pitch angle, yaw rotation speed, and throttle when moving are set to 30% of their maximum values.

## Record the reference path



