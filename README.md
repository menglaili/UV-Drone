# UV-Drone
Repository for UV-C Drone project

# Test data
Additional MP4 FPV video sequences with drone pose estimation can be downloaded [here](https://drive.google.com/drive/folders/10OSnv5N5SV3ehiEVNY__JBu7SiakYGfO?usp=sharing).

The pose estimation is from Webcam localization system with an error smaller than 10cm. To obtain the pose data, simply use

```
import numpy as np

pose_list = np.load('./pose_1.npy')
```
Each pose vector contains 3D position vector and orientation in quaternion i.e. [index, x, y, z, qw, qx, qy, qz]. There may be a small number of pose in the begining or end of the list not having correponding frames in the video.
