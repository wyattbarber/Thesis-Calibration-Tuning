import cv2
import numpy as np
from typing import List, Tuple, Any, Dict


class PinholeCalibrator:
    """
    Creates calibration data for intrinsics and distortion of a pinhole camera.

    :param List[str] files: List of file names to use for input images.
    """
    _min_points = 10

    _files: List[str] # List offile names
    _objpoints: List[cv2.Mat] # Object points of calibration pattern in each image
    _imgpoints: List[cv2.Mat] # Image points of calibration pattern in each image
    _rvecs: List[cv2.Mat] # Rotation of detected boards
    _tvecs: List[cv2.Mat] # Translation of detected boards

    _residuals: List[cv2.Mat] # Reprojection errors of each detected point
    _outliers: List[Tuple[int,int]] # Image and point indices of outliers
    _invalidated: int # Number of images invalidated due to too many outliers

    _intrinsic: cv2.Mat
    _distortion: cv2.Mat
    _rms: float

    def __init__(self, imsize: Tuple[int, int], boardsize: Tuple[int, int], squaresize: float, markersize: float, files: List[str]):
        self._imsize = imsize
        self._orig_count = len(files)

        self._dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._board = cv2.aruco.CharucoBoard(boardsize, squaresize, markersize, self._dict)
        self._detector = cv2.aruco.CharucoDetector(self._board)

        self._files = files
        self._objpoints = []
        self._imgpoints = []
        self._outliers = []
        for i, f in enumerate(files):
            self._load_image(i, f)

        self.recalculate()

    def _load_image(self, idx: int, filename: str):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        charuco_corners, charucos_ids, marker_corners, marker_ids = \
            self._detector.detectBoard(im)
        
        if (charucos_ids is not None) and (len(charucos_ids) >= self._min_points):
            # charuco_corners_refined = cv2.cornerSubPix(im, charuco_corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints, imgpoints = self._board.matchImagePoints(charuco_corners, charucos_ids)

            self._objpoints.append(objpoints)
            self._imgpoints.append(imgpoints)

    def _estimate(self):
        self._rms, self._intrinsic, self._distortion, self._rvecs, self._tvecs = cv2.calibrateCamera(
            self._objpoints, self._imgpoints, 
            self._imsize, None, None
            
            )

    def _calculate_residuals(self):
        for i in range(self.count_valid):
            imgpoints_est, _ = cv2.projectPoints(
                self._objpoints[i], self._rvecs[i], self._tvecs[i], self._intrinsic, self._distortion
            )
            self._residuals[i] = imgpoints_est - self._imgpoints[i]

    def remove_outliers(self, a: float):
        s = np.std(self.residuals)
        m = np.mean(self.residuals)

        idx_to_remove = []
        for i in range(self.count_valid):
            errs = self._residuals[i]
            outliers = []
            for j in range(errs.shape[0]):
                r = np.linalg.norm(errs[j,:]) - m
                if abs(r) > (a*s):
                    outliers.append(j)
            if (self._objpoints[i].shape[0] - len(outliers)) < self._min_points:
                idx_to_remove.append(i)
            else:
                self._objpoints[i] = np.delete(
                    self._objpoints[i], outliers, 0
                )
                self._imgpoints[i] = np.delete(
                    self._imgpoints[i], outliers, 0
                )
        for i in idx_to_remove:            
            self._objpoints.pop(i)
            self._imgpoints.pop(i)
        

    def recalculate(self):
        self._estimate()
        self._residuals = [None] * self.count_valid
        self._calculate_residuals()

    @property 
    def image_size(self):
        return self._imsize
    
    @property
    def board(self):
        return self._board
    
    @property
    def intrinsic(self):
        return self._intrinsic
    
    @property
    def distortion(self):
        return self._distortion
    
    @property
    def rms(self):
        return self._rms
    
    @property 
    def count(self):
        return self._orig_count
        
    @property 
    def count_valid(self):
        return len(self._imgpoints)
    
    @property
    def residuals(self):
        out = []
        for r in self._residuals:
            for i in range(r.shape[0]):
                err = np.linalg.norm(r[i,:])
                out.append(err)
        return out
    
    @property
    def residual_vecs(self):
        out = []
        for r in self._residuals:
            for i in range(r.shape[0]):
                out.append(r[i,:])
        return out
    
    def _test(self, file: str):
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        charuco_corners, charucos_ids, marker_corners, marker_ids = \
            self._detector.detectBoard(im)
        
        if (charucos_ids is not None) and (len(charucos_ids) > self._min_points):
            # charuco_corners_refined = cv2.cornerSubPix(im, charuco_corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints, imgpoints = self._board.matchImagePoints(charuco_corners, charucos_ids)

            _, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, self.intrinsic, self.distortion)

            imgpoints_est, _ = cv2.projectPoints(
                objpoints, rvec, tvec, self.intrinsic, self.distortion
            )
            return imgpoints_est - imgpoints
        
        else:
            return None
        
    def test(self, files: List[str]):
        out = []
        for file in files:
            err = self._test(file)
            if err is not None:
                for i in range(err.shape[0]):
                    out.append(err[i])
        return out
    
    @property
    def coverage(self):
        out = np.zeros((self.image_size[1], self.image_size[0]))
        for imgpts in self._imgpoints:
            cv2.aruco.drawDetectedCornersCharuco(out, imgpts)
        return out
    

class StereoCalibrator:
    """
    Creates a stereo calibration object.

    :param PinholeCalibrator left: Left camera calibration object.    
    :param PinholeCalibrator right: Right camera calibration object.
    :param List[Tuple[str, str]] files: Names of left, right image pairs to calibrate with.
    """
    _min_points = 6

    _left: PinholeCalibrator
    _right: PinholeCalibrator
    _files: List[Tuple[str, str]]

    _objpoints: List[cv2.Mat]
    _imgpoints_l: List[cv2.Mat]
    _imgpoints_r: List[cv2.Mat]
    _rvecs: List[cv2.Mat] # Rotation of detected boards in left camera frame
    _tvecs: List[cv2.Mat] # Translation of detected boards in left camera frame
    
    _left_residuals: List[cv2.Mat]
    _right_residuals: List[cv2.Mat]

    _rms: float
    _R: cv2.Mat
    _T: cv2.Mat
    _M1: cv2.Mat
    _M2: cv2.Mat
    _d1: cv2.Mat
    _d2: cv2.Mat

    def __init__(self, left: PinholeCalibrator, right: PinholeCalibrator, files: List[Tuple[str, str]]):
        self._files = files
        self._orig_count = len(files)
        self._left = left
        self._right = right
        self._board = left.board
        self._detector = cv2.aruco.CharucoDetector(self._board)

        self._objpoints = []
        self._imgpoints_l = []
        self._imgpoints_r = []
        self._tvecs = []
        self._rvecs = []

        for i, (l, r) in enumerate(files):
            self._load_image(i, l, r)
        self.recalculate()

    def _load_image(self, idx: int, file_left: str, file_right: str):
        iml = cv2.imread(file_left, cv2.IMREAD_GRAYSCALE)
        imr = cv2.imread(file_right, cv2.IMREAD_GRAYSCALE)


        charuco_corners_l, charuco_ids_l, marker_corners_l, marker_ids_l = \
            self._detector.detectBoard(iml)
        charuco_corners_r, charuco_ids_r, marker_corners_r, marker_ids_r = \
            self._detector.detectBoard(imr)
        
        if (charuco_corners_l is not None) and (len(charuco_corners_l) >= self._min_points) and \
            (charuco_corners_r is not None) and (len(charuco_corners_r) >= self._min_points):
            # charuco_corners_refined_l = cv2.cornerSubPix(
            #     iml, 
            #     charuco_corners_l, 
            #     (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # )
            objpoints_l, imgpoints_l = self._board.matchImagePoints(charuco_corners_l, charuco_ids_l)
            _, rv, tv = cv2.solvePnP(objpoints_l, imgpoints_l, self._left.intrinsic, self._left.distortion)
            
            # charuco_corners_refined_r = cv2.cornerSubPix(
            #     imr, 
            #     charuco_corners_r, 
            #     (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # )
            objpoints_r, imgpoints_r = self._board.matchImagePoints(charuco_corners_r, charuco_ids_r)
        
            objpoints_b, imgpoints_b_l, imgpoints_b_r = self._prune_ids(
                charuco_ids_l, objpoints_l, imgpoints_l, charuco_ids_r, objpoints_r, imgpoints_r
            )

            if len(objpoints_b) > self._min_points:
                self._tvecs.append(tv)
                self._rvecs.append(rv)
                self._objpoints.append(objpoints_b)
                self._imgpoints_l.append(imgpoints_b_l)
                self._imgpoints_r.append(imgpoints_b_r)

    def _prune_ids(self, idl, objl, imgl, idr, objr, imgr):
        id_both = np.intersect1d(idl, idr)
        objpoints_b = []
        imgpoints_b_l = []
        imgpoints_b_r = []

        for id in id_both:
            idxl = np.where(idl == id)[0][0]
            idxr = np.where(idr == id)[0][0]
            objpoints_b.append(objl[idxl])
            imgpoints_b_l.append(imgl[idxl])
            imgpoints_b_r.append(imgr[idxr])

        return np.array(objpoints_b), np.array(imgpoints_b_l), np.array(imgpoints_b_r)

    def _estimate(self):
        self._rms, self._M1, self._d1, self._M2, self._d2, self._R, self._T, _, _ = \
            cv2.stereoCalibrate(self._objpoints, self._imgpoints_l, self._imgpoints_r,
                self._left.intrinsic, self._left.distortion, 
                self._right.intrinsic, self._right.distortion, 
                self.image_size, flags = cv2.CALIB_FIX_INTRINSIC
            )

    def _calculate_residuals(self):
        for i in range(self.count_valid):
            imgpoints_est_l, _ = cv2.projectPoints(
                self._objpoints[i], self._rvecs[i], self._tvecs[i], self._M1, self._d1
            )
            self._left_residuals[i] = imgpoints_est_l - self._imgpoints_l[i]

            r, _ = cv2.Rodrigues(self._rvecs[i])
            r2 = self.R @ r
            r3, _ = cv2.Rodrigues(r2)
            t = self.R @ self._tvecs[i] + self.T
            imgpoints_est_r, _ = cv2.projectPoints(
                self._objpoints[i], r3, t, self._M2, self._d2
            )
            self._right_residuals[i] = imgpoints_est_r - self._imgpoints_r[i]

    def remove_outliers(self, a: float):
        sl = np.std([r[0] for r in self.residuals])
        ml = np.mean([r[0] for r in self.residuals])
        sr = np.std([r[1] for r in self.residuals])
        mr = np.mean([r[1] for r in self.residuals])

        idx_to_remove = []
        for i in range(self.count_valid):
            errs_l = self._left_residuals[i]
            errs_r = self._right_residuals[i]

            outliers = []
            for j in range(errs_l.shape[0]):
                rl = np.linalg.norm(errs_l[j,:]) - ml
                rr = np.linalg.norm(errs_r[j,:]) - mr
                if (abs(rl) > (a*sl)) or (abs(rr) > (a*sr)):
                    outliers.append(j)
            if (self._objpoints[i].shape[0] - len(outliers)) < self._min_points:
                idx_to_remove.append(i)
            else:
                self._objpoints[i] = np.delete(
                    self._objpoints[i], outliers, 0
                )
                self._imgpoints_l[i] = np.delete(
                    self._imgpoints_l[i], outliers, 0
                )
                self._imgpoints_r[i] = np.delete(
                    self._imgpoints_r[i], outliers, 0
                )
        for i in idx_to_remove:            
            self._objpoints.pop(i)
            self._imgpoints_l.pop(i)
            self._imgpoints_r.pop(i)
            self._tvecs.pop(i)
            self._rvecs.pop(i)

    def recalculate(self):
        self._estimate()
        self._left_residuals = [None] * self.count_valid
        self._right_residuals = [None] * self.count_valid
        self._calculate_residuals()

    @property 
    def image_size(self):
        return self._left.image_size
    
    @property
    def board(self):
        return self._board
    
    @property
    def intrinsic(self):
        return (self._M1, self._M2)
    
    @property
    def distortion(self):
        return (self._d1, self._d2)
    
    @property
    def R(self):
        return self._R

    @property    
    def T(self):
        return self._T
    
    @property
    def rms(self):
        return self._rms
    
    @property 
    def count(self):
        return self._orig_count
        
    @property 
    def count_valid(self):
        return len(self._objpoints)
    
    @property
    def residuals(self):
        out = []
        for i in range(len(self._left_residuals)):
            for j in range(self._left_residuals[i].shape[0]):
                err_l = np.linalg.norm(self._left_residuals[i][j,:])
                err_r = np.linalg.norm(self._right_residuals[i][j,:])
                out.append((err_l, err_r))
        return out
    
    @property
    def residual_vecs(self):
        out = []
        for i in range(len(self._left_residuals)):
            for j in range(self._left_residuals[i].shape[0]):
                out.append((
                    self._left_residuals[i][j,:], 
                    self._right_residuals[i][j,:]
                ))
        return out
    

class PinholeCrossValidator:

    def __init__(self, 
                 imsize: Tuple[int, int], boardsize: Tuple[int, int], squaresize: float, markersize: float, 
                 files: List[str], split: float = 0.7):
        idx_split = int(len(files) * split)
        np.random.shuffle(files)
        self._train = files[:idx_split]
        self._test = files[idx_split:]

        self._model = PinholeCalibrator(imsize, boardsize, squaresize, markersize, self._train)
        self._test_errors = self._model.test(self._test)

    @property
    def model(self):
        return self._model
    
    @property
    def count(self):
        return len(self._test)
    
    @property
    def rms(self):
        return np.mean(self.residuals)
    
    @property
    def residuals(self):
        return [np.linalg.norm(e) for e in self._test_errors]