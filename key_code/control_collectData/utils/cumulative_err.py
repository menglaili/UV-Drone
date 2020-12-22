import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from pyquaternion import Quaternion

def qua2rad(qua):
    return Quaternion(qua).yaw_pitch_roll[0]

def gt_rel_qua(gt_query, gt_base):
    gt_base_R = Quaternion(gt_base).normalised.rotation_matrix
    gt_query_R = Quaternion(gt_query).normalised.rotation_matrix
    gt_rel_R = np.linalg.inv(gt_base_R).dot(gt_query_R)
    gt_rel_qua = Quaternion(matrix=gt_rel_R).elements
    return gt_rel_qua

def gt_rel_translation(gt_query_trans, gt_base_qua, gt_base_trans):
    gt_base_R = Quaternion(gt_base_qua).normalised.rotation_matrix
    return np.linalg.inv(gt_base_R).dot(gt_query_trans - gt_base_trans)



def quat_abs_err(est_qua, gt_query, gt_base): # all input is numpy array
    est_rel_R = Quaternion(est_qua).normalised.rotation_matrix
    gt_base_R = Quaternion(gt_base).normalised.rotation_matrix
    est_query_R = gt_base_R.dot(est_rel_R)
    gt_query_R = Quaternion(gt_query).normalised.rotation_matrix
    diff_R = np.linalg.inv(est_query_R).dot(gt_query_R)
    return np.rad2deg(np.arccos((np.trace(diff_R)-1)/2))

def dis_abs(est_trans, gt_base_qua, gt_base_trans):
    gt_base_R = Quaternion(gt_base_qua).normalised.rotation_matrix
    est_query_trans = gt_base_R.dot(est_trans) + gt_base_trans
    return est_query_trans

def dis_abs_err(est_trans, gt_query_trans, gt_base_qua, gt_base_trans):
    est_query_trans = dis_abs(est_trans, gt_base_qua, gt_base_trans)
    return np.linalg.norm(est_query_trans - gt_query_trans)

def plot_print_result(result_list, name = "1.png", savefig = False, xlabel = ""):
    plt.figure()
    plt.hist(np.array(result_list), 30, density=1, cumulative = True, label = ("x", "y", "z"))
    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Cumulative Histogram')
    plt.tight_layout()
    if savefig:
        plt.savefig(name)
    print("mean: %1.4f; std: %1.4f" % (np.mean(result_list), np.std(result_list)))

def trans_error_plot_new_xyz(gt_translation, est_translation, savefig = False):
    tran_diff_xyz = []
    tran_diff_xyz_sc = []
    for i in range(gt_translation.shape[0]):
        err_distance_xyz = np.linalg.norm(gt_translation[i, :] - est_translation[i, :])
        err_distance_xyz_sc = gt_translation[i, :] - est_translation[i, :]
        tran_diff_xyz.append(err_distance_xyz)
        tran_diff_xyz_sc.append(err_distance_xyz_sc)
    plot_print_result(tran_diff_xyz, name="tran_diff_xyz.png", xlabel="Translation error(xyz)/m", savefig=savefig)
    plot_print_result(tran_diff_xyz_sc, name="tran_diff_xyz_sc.png", xlabel="Translation error(xyz)/m", savefig=savefig)
    return tran_diff_xyz

def trans_error_plot_new(gt_pose, est_translation, savefig = False):
    gt_query_translation = gt_pose[:, 7:10]
    gt_base_translation = gt_pose[:, :3]
    tran_diff_xyz = []
    tran_diff_zyx = []
    for i in range(gt_pose.shape[0]):
        gt_query_trans = gt_query_translation[i, :]
        est_trans = est_translation[i, :]
        gt_base_trans = gt_base_translation[i, :]
        err_distance_xyz = np.linalg.norm(est_trans + gt_base_trans - gt_query_trans)
        err_distance_zyx = np.linalg.norm(np.flip(est_trans) + gt_base_trans - gt_query_trans)
        tran_diff_xyz.append(err_distance_xyz)
        tran_diff_zyx.append(err_distance_zyx)
    plot_print_result(tran_diff_zyx, name="tran_diff_zyx.png", xlabel="Translation error(zyx)/m", savefig=savefig)
    plot_print_result(tran_diff_xyz, name="tran_diff_xyz.png", xlabel="Translation error(xyz)/m", savefig=savefig)
    return tran_diff_xyz, tran_diff_zyx

def trans_error_plot(gt_pose, est_translation, savefig = False):
    gt_base_orientation = gt_pose[:, 3:7]
    gt_query_translation = gt_pose[:, 7:10]
    gt_base_translation = gt_pose[:, :3]
    tran_diff_xyz = []
    tran_diff_zyx = []
    for i in range(gt_pose.shape[0]):
        gt_base_qua = gt_base_orientation[i, :]
        gt_query_trans = gt_query_translation[i, :]
        est_trans = est_translation[i, :]
        gt_base_trans = gt_base_translation[i, :]
        err_distance_xyz = dis_abs_err(est_trans, gt_query_trans, gt_base_qua, gt_base_trans)
        err_distance_zyx = dis_abs_err(np.flip(est_trans), gt_query_trans, gt_base_qua, gt_base_trans)
        tran_diff_xyz.append(err_distance_xyz)
        tran_diff_zyx.append(err_distance_zyx)
    plot_print_result(tran_diff_zyx, name="tran_diff_zyx.png", xlabel = "Translation error(zyx)/m", savefig = savefig)
    plot_print_result(tran_diff_xyz, name="tran_diff_xyz.png", xlabel = "Translation error(xyz)/m", savefig = savefig)
    return tran_diff_xyz, tran_diff_zyx

def qua_error_plot(gt_pose, est_orientation, savefig = False):
    gt_query_orientation = gt_pose[:, 3:7]
    gt_base_orientation = gt_pose[:, 10:]

    qua_diff = []
    for i in range(gt_pose.shape[0]):
        est_qua = est_orientation[i, :]
        gt_query_qua = gt_query_orientation[i, :]
        gt_base_qua = gt_base_orientation[i, :]
        err_angle = quat_abs_err(est_qua, gt_query_qua, gt_base_qua)
        qua_diff.append(err_angle)
    plot_print_result(qua_diff, name="qua_diff.png", xlabel = "Orientation error/deg", savefig = savefig)
    return qua_diff

if "__name__" == "__main__":
    with open("./utils/res.bin", 'rb') as fid:
        data_array = np.fromfile(fid, np.float32).reshape((-1, 7)).T  ## [qw, qx, qy, qz, x, y, z].T
    est_orientation = data_array[:4, :].T
    est_translation = data_array[4:, :].T
    # the post pose is the database, the pre post is the query one, relative pose is from frame database to query
    gt_pose = np.loadtxt("./data/test_pose.txt")

    tran_diff_xyz, tran_diff_zyx = trans_error_plot_new(gt_pose, est_translation)
    qua_error_plot(gt_pose, est_orientation)