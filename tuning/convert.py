import numpy as np
from glob import glob
import os
from copy import deepcopy
from scipy.spatial.transform import Rotation
import re
import struct
import cv2
import h5py


class Data:
    def __init__(self, file: str):        
        with h5py.File(file) as data:
            self._video = data["video"][()]
            self._depth = data["depth"][()]
            self._v = data["velocity"][()]
            self._w = data["angular_velocity"][()]
            self._ts = data["timestamps"][()]     
        self._crop = -1
        fname = os.path.basename(file)
        id, _ = os.path.splitext(fname)
        self._id = id

    def crop(self, c: float):
        """
        Returns a cropped deep copy of the data

        Cropping is done as a percentage of the original length, always starting at index 0.

        :param float c: Amount of the original data to keep, in %.

        :return: Cropped data copy
        :rtype: Data
        """
        out = deepcopy(self)
        out._crop = int(c * len(self))
        return out

    def __len__(self):
        """
        :return: Number of samples recorded in this dataset.
        :rtype: int
        """
        return self.ts.shape[0]
    
    @property
    def video(self):
        return self._video[:self._crop,...]
    
    @property
    def depth(self):
        return self._depth[:self._crop,...]
    
    @property
    def velocity(self):
        return self._v[:self._crop,...]
    
    @property
    def angular_velocity(self):
        return self._w[:self._crop,...]
    
    @property
    def accel(self):
        return None
    
    @property
    def gyro(self):
        return None
    
    @property
    def ts(self):
        return self._ts[:self._crop,...]
    
    @property
    def id(self):
        return self._id
    
SIZE = (600, 800)

def load_depth(path: str, idx: int):
    file = os.path.join(path, f"frame{idx}.txt")    
    with open(file, "rb") as src:
        data = src.read()
    return np.ndarray(SIZE, np.float32, data)

def load_image(path: str, idx: int):
    file = os.path.join(path, f"frame{idx}.txt")    
    with open(file, "rb") as src:
        data = src.read()
    return np.ndarray((*SIZE, 3), np.uint8, data)


def _load_pose(file: str, ts: np.ndarray):
    d = []
    a = []
    t = []
    with open(file, "r") as f:
        for line in f.readlines():
            pose, _t = line.split('@')
            t.append(float(_t))
            nums = pose.split()
            d.append(np.array([float(x) for x in nums[:3]]))
            a.append(np.array([float(x) for x in nums[3:]]))

    N = len(d)
    v = np.zeros((N-1,3))
    w = np.zeros((N-1,3))
    for i in range(N-1):
        r = Rotation.from_euler("xyz", (a[i] + a[i+1])/2.0).as_matrix()
        v[i,:] = np.matmul(r.transpose(), ((d[i+1] - d[i]) / (t[i+1] - t[i])))
        w[i,:] = np.matmul(r.transpose(), ((a[i+1] - a[i]) / (t[i+1] - t[i])))
    
    M = len(ts)
    vi = np.zeros((M,3))
    wi = np.zeros((M,3))
    for i in range(3):
        vi[:,i] = np.interp(ts, t[:-1], v[:,i])
        wi[:,i] = np.interp(ts, t[:-1], w[:,i])
    
    return vi, wi


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    for dir in glob(os.path.join(root, "raw", "*")):
        print(f"Loading from {dir}")
        depth = _load_depth(os.path.join(dir, "depth"))
        video = _load_video(os.path.join(dir, "testvideo.mp4"))
        ts = np.array([i/30 for i in range(video.shape[0])])
        v, w = _load_pose(os.path.join(dir, "pose.txt"), ts)
        id = os.path.basename(os.path.normpath(dir))

        
        print(f"Saving {id}")
        with h5py.File(os.path.join(root, "preprocessed", f"{id}.h5") , "w") as newdata:
            newdata.create_dataset("video", data=video, dtype=np.uint8)
            newdata.create_dataset("depth", data=depth)
            newdata.create_dataset("velocity", data=v)
            newdata.create_dataset("angular_velocity", data=w)
            newdata.create_dataset("timestamps", data=ts)

            