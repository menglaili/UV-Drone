import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import os



Output_root = './train_data_complete'
Data_root = './data'

plt.figure(figsize=(8,5))
plt.ion()
plt.show()
while True:
    if os.path.isfile(os.path.join(Output_root, "cur_count.txt")):
        cur_count = np.loadtxt(os.path.join(Output_root, "cur_count.txt")).astype(int)
        img_ref_fig = plt.imread(os.path.join(Data_root,'image',str(cur_count)+'.png'))
        # cv2.imshow(str(cur_count), img_ref_fig)
        # cv2.waitKey(10000)
        plt.imshow(img_ref_fig)
        plt.draw()
        plt.pause(0.001)
        # plt.ioff()
        # plt.show(block = False)  #