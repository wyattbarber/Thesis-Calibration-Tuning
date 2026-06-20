import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import open3d
import sys

N = 3

with open("parameters.pickle", "rb") as file:
    M1, d1, M2, d2, T, R, Q, ROI1, ROI2, R1, R2, Pn1, Pn2 = pickle.load(file)
mapl1, mapl2 = cv2.initUndistortRectifyMap(M1, d1, R1, Pn1, (800,600), cv2.CV_32F)
mapr1, mapr2 = cv2.initUndistortRectifyMap(M2, d2, R2, Pn2, (800,600), cv2.CV_32F)

left = cv2.remap(
    cv2.imread(f"left-{N}.png", cv2.IMREAD_GRAYSCALE),
    mapl1, mapl2,
    cv2.INTER_LINEAR
    )
right = cv2.remap(
    cv2.imread(f"right-{N}.png", cv2.IMREAD_GRAYSCALE),
    mapr1, mapr2,
    cv2.INTER_LINEAR
    )

cv2.namedWindow('d',cv2.WINDOW_NORMAL)
cv2.resizeWindow('d',600,600)

def nothing(x):
    pass

cv2.createTrackbar('OutputDepth','d',0,1,nothing)

cv2.createTrackbar('sgbm','d',0,1,nothing)
cv2.createTrackbar('equalize','d',0,1,nothing)

cv2.createTrackbar('numDisparities','d',1,17,nothing)
cv2.createTrackbar('blockSize','d',5,50,nothing)
cv2.createTrackbar('preFilterType','d',1,1,nothing)
cv2.createTrackbar('preFilterSize','d',2,25,nothing)
cv2.createTrackbar('preFilterCap','d',5,62,nothing)
cv2.createTrackbar('textureThreshold','d',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','d',15,100,nothing)
cv2.createTrackbar('speckleRange','d',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','d',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','d',5,25,nothing)
cv2.createTrackbar('minDisparity','d',5,25,nothing)

cv2.createTrackbar('lambda','d',1000,50000,nothing)
cv2.createTrackbar('sigma','d',50,1000,nothing)
cv2.createTrackbar('discontinuityRadius','d',1,15,nothing)
cv2.createTrackbar('LRCthresh','d',1,15,nothing)


while True:

    try:
        depth_view = bool(cv2.getTrackbarPos("OutputDepth", 'd'))

        sgbm = bool(cv2.getTrackbarPos("sgbm", 'd'))
        equalize = bool(cv2.getTrackbarPos("equalize", 'd'))

        numDisparities = cv2.getTrackbarPos('numDisparities','d')*16
        blockSize = cv2.getTrackbarPos('blockSize','d')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','d')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','d')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','d')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','d')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','d')
        speckleRange = cv2.getTrackbarPos('speckleRange','d')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','d')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','d')
        minDisparity = cv2.getTrackbarPos('minDisparity','d')

        filterLambda = cv2.getTrackbarPos('lambda','d')
        filterSigma = cv2.getTrackbarPos('sigma','d')/100
        filterDR = cv2.getTrackbarPos('discontinuityRadius','d')
        filterLRC = cv2.getTrackbarPos('LRCthresh','d')

        stereo = cv2.StereoSGBM_create() if sgbm else cv2.StereoBM_create()
        if not sgbm:
            stereo.setPreFilterType(preFilterType)
            stereo.setPreFilterSize(preFilterSize)
            stereo.setTextureThreshold(textureThreshold)
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        stereo_r = cv2.ximgproc.createRightMatcher(stereo)
        filter = cv2.ximgproc.createDisparityWLSFilter(stereo)

        l = cv2.equalizeHist(left) if equalize else left
        r = cv2.equalizeHist(right) if equalize else right

        dl = stereo.compute(l, r)    
        dr = stereo_r.compute(r, l)
        filter.setLambda(filterLambda)
        filter.setSigmaColor(filterSigma)
        filter.setDepthDiscontinuityRadius(filterDR)
        filter.setLRCthresh(filterLRC)
        disparity = filter.filter(dl, left, None, dr)
        # disparity = dl

        disparity = disparity.astype(np.float32)
        disparity/=16.0

        if not depth_view:
            disparity = (disparity - minDisparity)/numDisparities

            cv2.imshow("d", disparity)

        else:
            Pmax = Q @ np.array([
                [0],
                [0],
                [stereo.getMinDisparity()],
                [1]
            ])
            Pmax /= Pmax[3]
            Dmax = np.linalg.norm(Pmax[:3])
            pt_cloud = cv2.reprojectImageTo3D(disparity, Q)
            depth = pt_cloud[:,:,2]
            depth[depth < 0] = 0
            depth[depth > Dmax] = Dmax
            depth /= np.max(depth)

            cv2.imshow("d", 1.0 - depth)

        with open(sys.argv[1], "w") as file:
            file.writelines([
                "{",
                f"    \"numDisparities\": {numDisparities},",
                f"    \"blockSize\": {blockSize},",
                f"    \"preFilterType\": {preFilterType},",
                f"    \"preFilterSize\": {preFilterSize},",
                f"    \"preFilterCap\": {preFilterCap},",
                f"    \"textureThreshold\": {textureThreshold},",
                f"    \"speckleRange\": {speckleRange},",
                f"    \"speckleWindowSize\": {speckleWindowSize},",
                f"    \"disp12MaxDiff\": {disp12MaxDiff},",
                f"    \"uniquenessRatio\": {uniquenessRatio},",
                f"    \"minDisparity\": {minDisparity},",
                f"    \"filterLambda\": {filterLambda},",
                f"    \"filterSigma\": {filterSigma},",
                f"    \"filterDR\": {filterDR},",
                f"    \"filterLRC\": {filterLRC}",
                "}"
            ])
        
        if cv2.waitKey(1) == 27:
            break
    except KeyboardInterrupt:
        break