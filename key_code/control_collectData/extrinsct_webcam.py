import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math


def undistort_img(img):
    mtx = cam_dic["intrinsic_matrix"]
    dist = cam_dic["distortion_coefficients"]
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def tohomo(point_pixel):
    r = len(point_pixel)
    a = np.zeros(r+1)
    a[-1] = 1
    a[:-1] = point_pixel
    return a

def decomposeRT(trans_mat):
    R = trans_mat[:3, :3]
    t = trans_mat[:3, 3]
    return R,t

def combineRT(R,t):
    trans_mat = np.zeros((4,4))
    trans_mat[3,3] = 1
    trans_mat[:3,:3] = R
    trans_mat[:3,3] = t
    return trans_mat

def homo2(pose):
    Roc = np.zeros((3, 4))
    Roc[:, :3] = np.eye(3)
    z0 = pose[2]
    return (1/z0)*Roc.dot(pose)

with open('./camera_extrinst_plot_errorbar/results.pickle', 'rb') as f:
    cam_dic = pickle.load(f)
camera_matrix = cam_dic["intrinsic_matrix"]
trans_mat = np.load('./camera_extrinst_plot_errorbar/trans_mat_scene1.npy')
def pixel2world(W, point_pixel):
    p1 = tohomo(point_pixel)
    p2 = np.linalg.inv(camera_matrix).dot(p1)
    k = p2[0]
    l = p2[1]
    R,t = decomposeRT(trans_mat)
    R0 = np.zeros((4,4))
    R0[1:,1:] = R
    R = R0
    A = np.array([[k, 0],[0, l]]).dot(np.array([[R[3,1], R[3,2]],[R[3,1], R[3,2]]])) - np.array([[R[1,1], R[1,2]],[R[2,1], R[2,2]]])
    b = np.zeros((2,1))
    b[0] = (R[1,3]-k*R[3,3])*W + t[0] - k*t[2]
    b[1] = (R[2,3]-l*R[3,3])*W + t[1] - l*t[2]
    point_world = np.linalg.solve(A, b)
    point_world = np.hstack((point_world[:,0], W))
    return point_world