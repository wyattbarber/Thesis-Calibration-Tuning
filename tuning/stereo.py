import numpy as np
import cv2
from convert import SIZE
import pickle

with open("parameters.pickle", "rb") as file:
    M1, d1, M2, d2, T, R, Q, ROI1, ROI2, R1, R2, Pn1, Pn2 = pickle.load(file)
ML1, ML2 = cv2.initUndistortRectifyMap(M1, d1, R1, Pn1, (SIZE[1], SIZE[0]), cv2.CV_32FC1)
MR1, MR2 = cv2.initUndistortRectifyMap(M2, d2, R2, Pn2, (SIZE[1], SIZE[0]), cv2.CV_32FC1)

def crop(im, rect):
    return im[
        rect[1]:rect[1]+rect[3],
        rect[0]:rect[0]+rect[2]
    ]


def compute_depth(left, right):
    left_rect = cv2.remap(left, ML1, ML2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, MR1, MR2, cv2.INTER_LINEAR)

    bm = cv2.StereoSGBM.create(minDisparity=0, numDisparities=16*4, blockSize=5)
    bm.setP1(8 * bm.getBlockSize()**2)
    bm.setP2(64 * bm.getBlockSize()**2)
    # bm.setDisp12MaxDiff(1)
    # bm.setUniquenessRatio(5)
    # bm.setSpeckleWindowSize(50)
    # bm.setSpeckleRange(1)
    # bm.setMode(cv2.STEREO_SGBM_MODE_HH)

    disp = bm.compute(left_rect, right_rect)
    # return disp
    pt_cloud = cv2.reprojectImageTo3D((disp / 16).astype(np.float32), Q)
    return np.linalg.norm(pt_cloud, axis=2)
