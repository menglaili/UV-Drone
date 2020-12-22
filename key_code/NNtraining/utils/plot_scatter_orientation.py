from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import os

path0 = [(0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)), (0.5, 0), (0.5, 0)]
path1 = [(0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
     (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
      (0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0)]
# path2 = [(0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
#      (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
#       (0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
#        (0.5, 0), (0.5, 0), (0, np.radians(-90))]
# path2 = [(0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
#      (0, 0, 0.5), (1, 0), (0, np.radians(-90)),\
#       (1, 0), (0, 0, -0.5), (0.5, 0), (0, np.radians(-90)),\
#       (0.5, 0), (0.5, 0), (0, np.radians(-90))]

path_scene1 = [(0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
        (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
        (0.5, 0), (0.5, 0), (0.5, 0), (0.5, 0), (0, np.radians(-90)),\
        (0.5, 0), (0.5, 0), (0, np.radians(-90))]

def plot2D(data_2d1, data_2d2 = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(data_2d1[:, 0], data_2d1[:, 1], label="Open-loop", color = "black")
    print(len(data_2d2.tolist()))
    if len(data_2d2.tolist()) != 0:
        ax.plot(data_2d2[:, 0], data_2d2[:, 1], label="NN-corr", color="red")
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    return fig, ax

def plot3D(data_3d):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], label="Test", color = "red")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return fig, ax

def gt_rel_qua(gt_query, gt_base):
    gt_base_R = Quaternion(gt_base).normalised.rotation_matrix
    gt_query_R = Quaternion(gt_query).normalised.rotation_matrix
    gt_rel_R = np.linalg.inv(gt_base_R).dot(gt_query_R)
    gt_rel_qua = Quaternion(matrix=gt_rel_R).elements
    return gt_rel_qua



def quat_abs_err(est_qua, gt_query, gt_base): # all input is numpy array
    est_rel_R = Quaternion(est_qua).normalised.rotation_matrix
    gt_base_R = Quaternion(gt_base).normalised.rotation_matrix
    est_query_R = gt_base_R.dot(est_rel_R)
    gt_query_R = Quaternion(gt_query).normalised.rotation_matrix
    diff_R = np.linalg.inv(est_query_R).dot(gt_query_R)
    return np.rad2deg(np.arccos((np.trace(diff_R)-1)/2))

# def path2world(path):
#     cur_xy = np.array((0.0, 0.0))
#     turning = 1
#     radians = 0
#     track_list = [np.hstack((np.copy(cur_xy), 1, radians))]
#     for forward, turn in path:
#         if turn != 0:
#             turning = 1 - turning
#         if turning:
#             cur_xy[0] += int(np.cos(radians)) * forward
#         else:
#             cur_xy[1] += -int(np.sin(radians)) * forward
#         radians += turn
#         track_list.append(np.hstack((np.copy(cur_xy), 1, radians)))
#     track_list = np.array(track_list)
#     return track_list

def path2world(path):
    cur_xy = np.array((0.0, 0.0))
    turning = 1
    radians = 0
    height = 1
    track_list = [np.hstack((np.copy(cur_xy), 1, radians))]
    for _, data in enumerate(path):
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
        track_list.append(np.hstack((np.copy(cur_xy), height, radians)))
    track_list = np.array(track_list)
    return track_list
    # cur_xy = np.array((0.0, 0.0))
    # turning = 1
    # radians = 0
    # height = 1
    # track_list = [np.hstack((np.copy(cur_xy), 1, radians))]
    # for _, data in enumerate(path):
    #     if len(data) == 2:
    #         forward, turn, dh = data[0], data[1], 0
    #     elif len(data) == 3:
    #         forward, turn, dh = data[0], data[1], data[2]
    #     if turn != 0:
    #         turning = 1 - turning
    #     if turning:
    #         cur_xy[0] += int(np.cos(radians)) * forward
    #     else:
    #         cur_xy[1] += -int(np.sin(radians)) * forward
    #     radians += turn
    #     height += dh
    #     track_list.append(np.hstack((np.copy(cur_xy), height, radians)))
    # track_list = np.array(track_list)
    # return track_list

def plot3D_gt(data_3d, path = path1):
    fig, ax = plot3D(data_3d)
    gt_pose = path2world(path)
    ax.plot(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], label="Ground Truth")
    ax.legend()

def plot2D_gt(data_2d1, data_2d2 = None, path = path_scene1, title = "Average pose correction"):
    fig, ax = plot2D(data_2d1, data_2d2 = data_2d2)
    gt_pose = path2world(path)
    print(gt_pose)
    ax.plot(gt_pose[:, 0], gt_pose[:, 1], "-o", label="Ground Truth")
    ax.legend()
    ax.set_title(title)

# filled error bar
def plotfillederr(mean_pose, std_pose):
    cpind = np.arange(mean_pose.shape[0])
    clrs = ["blue", "red"]
    fig, ax = plt.subplots()
    ax.plot(cpind, mean_pose[:, 0], "-", c=clrs[0], label = "X-coordinate")
    ax.plot(cpind, mean_pose[:, 1], "-", c=clrs[1], label = "Y-coordinate")
    ax.fill_between(cpind, mean_pose[:, 0] - std_pose[:, 0], mean_pose[:, 0] + std_pose[:, 0], alpha=0.3, facecolor=clrs[0])
    ax.fill_between(cpind, mean_pose[:, 1] - std_pose[:, 1], mean_pose[:, 1] + std_pose[:, 1], alpha=0.3, facecolor=clrs[1])
    ax.set_xlabel('Indices')
    ax.set_ylabel('Distance/m')
    ax.set_title('Pose correction distribution')
    ax.legend()
    plt.tight_layout()

#
# gt_pose = path2world(path1)
# fig, ax = plot2D(gt_pose[:, :2])
# def plot2D_input(data_2d):
#     ax.plot(data_2d[:, 0], data_2d[:, 1], label="Test")

def plot2Derrbar(pose_berr_array, pose_aerr_array, ind_list):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    pose_berr_array = np.around(pose_berr_array, 3)
    pose_aerr_array = np.around(pose_aerr_array, 3)
    num = pose_berr_array.shape[0]
    xind = np.arange(num)
    labels = []
    for i in ind_list:
        labels.append("CP%d" % i)
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 0.8 # the width of the bars
    ax.bar(xind - 3*width/8, pose_berr_array[:, 0], width/4, label='X-axis' + " " + "before")
    ax.bar(xind - width/8, pose_aerr_array[:, 0], width/4, label='X-axis' + " " + "after")
    ax.bar(xind + width/8, pose_berr_array[:, 1], width/4, label='Y-axis' + " " + "before")
    ax.bar(xind + 3*width/8, pose_aerr_array[:, 1], width/4, label='Y-axis' + " " + "after")
    ax.set_ylabel('distance/m')
    ax.set_xticks(xind)
    ax.set_xticklabels(labels)
    ax.legend()
    # autolabel(rects1)
    # autolabel(rects2)
    fig.tight_layout()

def plotxyzerrbar(gt_list, est_translation):
    err_mse = (np.abs(est_translation - gt_list)).mean(axis=0)
    print(err_mse)
    xind = 0
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 0.6  # the width of the bars
    ax.bar(xind - 2*width / 3, err_mse[0], width / 3, label="X-axis")
    ax.bar(xind, err_mse[1], width / 3, label='Y-axis')
    ax.bar(xind + 2*width / 3, err_mse[2], width / 3, label='Z-axis')
    ax.set_ylabel('distance/m')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax.legend()
    fig.tight_layout()


def plot2Derrbar2(pose_berr_array, pose_aerr_array, pose_pred_array, ind_list):
    pose_berr_array = np.around(pose_berr_array, 3)
    pose_aerr_array = np.around(pose_aerr_array, 3)
    err_mse = np.around(pose_pred_array[:, :3], 3)
    num = pose_berr_array.shape[0]
    xind = np.arange(num)
    labels = []
    for i in ind_list:
        labels.append("CP%d" % i)
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 0.72 # the width of the bars
    ax.bar(xind - 3*width/6, pose_berr_array[:, 0], width/6, label='X-axis' + " " + "before")
    ax.bar(xind - 2*width/6, pose_aerr_array[:, 0], width/6, label='X-axis' + " " + "after")
    ax.bar(xind - 1*width/6, err_mse[:, 0], width/6, label='X-axis' + " ctrl")
    ax.bar(xind, pose_berr_array[:, 1], width / 6, label='Y-axis' + " " + "before")
    ax.bar(xind + 1*width/6, pose_aerr_array[:, 1], width/6, label='Y-axis' + " " + "after")
    ax.bar(xind + 2 * width / 6, err_mse[:, 1], width / 6, label='Y-axis' + " ctrl")
    ax.set_ylabel('distance/m')
    ax.set_xticks(xind)
    ax.set_xticklabels(labels)
    ax.legend()
    # autolabel(rects1)
    # autolabel(rects2)
    fig.tight_layout()

if "__name__" == "__main__":
    data_3d = np.loadtxt("NN_pose.txt")
    qua_array = data_3d[:, 3:7]
    fig = plt.figure()
    yaw_list = []
    # test the orientation correctness
    for i in range(data_3d.shape[0]):
        yaw = Quaternion(data_3d[i, 3:7]).yaw_pitch_roll[0]
        yaw = np.degrees(yaw)
        yaw_list.append(yaw)
        plt.plot(i, yaw, "o")
    plt.xlabel("index")
    plt.ylabel("degree")
    yaw_qua_array = np.hstack((qua_array, np.array(yaw_list).reshape(-1,1)))

    # test ground truth
    gt_err = np.zeros(data_3d.shape[0]-1)
    for i in range(data_3d.shape[0]-1):
        gt_qua = gt_rel_qua(yaw_qua_array[i, :-1], yaw_qua_array[i+1, :-1])
        gt_err[i] = quat_abs_err(gt_qua, yaw_qua_array[i, :-1], yaw_qua_array[i+1, :-1])

    plt.figure()
    plt.hist(np.array(gt_err), 30, density=1, cumulative = True)
    plt.xlabel('Orientation error/deg')
    plt.ylabel('Probability')
    plt.title('Cumulative Histogram')
    plt.tight_layout()



    # plot 3D
    data_3d = np.loadtxt("NN_pose.txt")
    plot3D(data_3d)


    ## plot 10 times trajectory of distance control
    from pyquaternion import Quaternion
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15})
    import os




    data_base_dir = "./trajectory_data"
    gt_pose = path2world(path1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2], label="Ground Truth")
    ax.legend()

    CorrTime = len(os.listdir(data_base_dir)) - 1
    CheckNum = 13#len(os.listdir(os.path.join(data_base_dir, str(0), "image")))
    mse_list = []
    mse_xyz_list = []
    cur_pose_list = []
    cur_pose_diff = np.zeros_like(gt_pose[:, :3])
    mse_x = []
    mse_y = []
    mse_z = []
    qua_average = 0
    yaw_list = []
    for i in range(CorrTime):
        file_dir = os.path.join(data_base_dir, str(i))
        cur_pose = np.loadtxt(os.path.join(file_dir, "pose.txt"))[:CheckNum, :3]
        ax.plot(cur_pose[:, 0], cur_pose[:, 1], cur_pose[:, 2], label="Trajactory%s" % str(i))
        mse_list.append(np.mean(np.sqrt(np.sum((gt_pose[:-4, :3] - cur_pose[:-4, :])**2, axis = 1))))
        cur_pose_list.append(cur_pose)
        mse_x.append(np.mean(abs(gt_pose[:, :3] - cur_pose)[:, 0]))
        mse_y.append(np.mean(abs(gt_pose[:, :3] - cur_pose)[:, 1]))
        mse_z.append(np.mean(abs(gt_pose[:, :3] - cur_pose)[:, 2]))
        cur_pose_diff += np.abs(gt_pose[:, :3] - cur_pose)
        cur_pose_all = np.loadtxt(os.path.join(file_dir, "pose.txt"))
        qua_average += np.abs(cur_pose_all[0, -4:])
        print((Quaternion(cur_pose_all[i, -4:]).yaw_pitch_roll[0] - np.radians(64.903)))
        yaw_list.append(np.abs(gt_pose[i, 3] - (Quaternion(cur_pose_all[i, -4:]).yaw_pitch_roll[0] - np.radians(64.903))))
    print("mean yaw", np.mean(yaw_list))
    print("std yaw", np.std(yaw_list))
    qua_average /= (i + 1)
    print("average ori ", np.degrees(Quaternion(qua_average).yaw_pitch_roll[0]))
    cur_pose_diff /= (i + 1)  # for each checkpoint what is the average of x,y,z abs difference
    minind = np.argmin(mse_list)
    mincur_pose = cur_pose_list[minind]
    cur_pose_list = np.array(cur_pose_list)
    mean_pose = np.mean(cur_pose_list, axis = 0)[:, :2]
    std_pose = np.std(cur_pose_list, axis = 0)[:, :2]
    # ax.plot(mincur_pose[:, 0], mincur_pose[:, 1], mincur_pose[:, 2], label="Trajactory%s" % str(minind))
    # ax.legend()
    # ax.set_zlim3d(0.5, 1.55)
    print(np.mean(mse_list), np.std(mse_list))
    print(np.mean(mse_x), np.mean(mse_y), np.mean(mse_z), np.std(mse_x), np.std(mse_y), np.std(mse_z))

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_zlabel('Z/m')
    fig.tight_layout()
    fig.savefig("trajectory.png")

    ## no control trajectory
    plotfillederr(cur_pose_list[3, :, :2], std_pose)
    plot2D_gt(cur_pose_list[3, :, :2], title="Average pose correction")