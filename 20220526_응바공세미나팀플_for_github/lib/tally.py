import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == "__main__":
    from fluka import USRBIN
else:
    from lib.fluka import USRBIN


def checkTupleIdentical(_tuple1, _tuple2):
    if len(_tuple1) != len(_tuple2):
        raise Exception()
    for i in range(len(_tuple1)):
        if _tuple1[i] != _tuple2[i]:
            raise Exception()


def meshtal(file_name, shape):  # mcnp MESHTAL
    with open(file_name) as file:
        lines = file.readlines()
        mesh = np.empty(len(lines), dtype=np.float64)
        for index, line in enumerate(lines):
            parsed = line.split()
            mesh[index] = float(parsed[3])

    return np.reshape(mesh, shape)


def usrbin(file_name, shape):  # fluka USRBIN
    fk = USRBIN(file_name)
    _ = fk.Get()
    _ = fk.Get()
    data_byte = fk.Get()

    fk_1d = np.frombuffer(data_byte, dtype=np.float32)
    arr_fk = fk_1d.reshape((shape[2], shape[1], shape[0]))

    return np.transpose(arr_fk, axes=(2,1,0))


def gpuMesh(name, index, nbatch, shape):
    arr_gpu = np.zeros(shape, dtype=np.float32)
    for i in range(nbatch):
        dtemp = np.load("{}{}_batch{}.npy".format(name, index, i))
        arr_gpu += dtemp

    return arr_gpu / nbatch


class Tally:
    def __init__(self, arr, origin, size):
        self._arr = arr
        self._origin = np.array(origin)
        self._size = np.array(size)
        self._has_ct = False

    def setDICOM(self, arr, origin, size):
        self._has_ct = True
        self._ct_arr = arr
        self._ct_origin = np.array(origin)
        self._ct_size = np.array(size)

    @staticmethod
    def _cropThreeDimArr(axis, position, arr, origin, size):
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        arr_index = int((position - origin[axis])/size[axis])
        img = None
        extent = [None, None, None, None]
        xx = None
        yy = None

        if axis == 0:
            img = arr[arr_index,:,:]
            xx = 2
            yy = 1
        elif axis == 1:
            img = np.transpose(arr[:,arr_index,:])
            xx = 0
            yy = 2
        elif axis == 2:
            img = arr[:,:,arr_index]
            xx = 1
            yy = 0

        extent[0] = origin[xx]
        extent[1] = origin[xx] + arr.shape[xx] * size[xx]
        extent[2] = origin[yy]
        extent[3] = origin[yy] + arr.shape[yy] * size[yy]

        return img, extent
    
    def getMaxValue(self, axis, position):
        tally_img, extent = self._cropThreeDimArr(axis, position, self._arr, self._origin, self._size)
        return np.max(tally_img)

    def plot(self, axis, position, norm="log", vmax=None, vmin=None):
        alpha = 1.0
        if self._has_ct:
            ct_img, extent = self._cropThreeDimArr(axis, position, self._ct_arr, self._ct_origin, self._ct_size)
            plt.imshow(ct_img, cmap="gray", extent=extent, origin='lower')
            alpha = 0.7

        tally_img, extent = self._cropThreeDimArr(axis, position, self._arr, self._origin, self._size)
        
        if norm == "log":
            plt.imshow(tally_img, cmap="jet", norm=LogNorm(vmin, vmax), extent=extent, alpha=alpha, origin='lower')
        else:
            plt.imshow(tally_img, cmap="jet", vmin=vmin, vmax=vmax, extent=extent, alpha=alpha, origin='lower')
        
        if axis == 0:
            plt.xlabel("Z position (cm)")
            plt.ylabel("Y position (cm)")
        elif axis == 1:
            plt.xlabel("X position (cm)")
            plt.ylabel("Z position (cm)")
        elif axis == 2:
            plt.xlabel("Y position (cm)")
            plt.ylabel("X position (cm)")
        
    def __add__(self, _tally):
        if type(_tally) != Tally:
            raise TypeError()
        checkTupleIdentical(self._origin, _tally._origin)
        checkTupleIdentical(self._size, _tally._size)

        arr_new = self._arr + _tally._arr
        return Tally(arr_new, self._origin, self._size)
    
    def __sub__(self, _tally):
        if type(_tally) != Tally:
            raise TypeError()
        checkTupleIdentical(self._origin, _tally._origin)
        checkTupleIdentical(self._size, _tally._size)

        arr_new = self._arr - _tally._arr
        return Tally(arr_new, self._origin, self._size)

    def __mul__(self, _tally):
        if type(_tally) != Tally:
            arr_new = self._arr * _tally
        else:
            checkTupleIdentical(self._origin, _tally._origin)
            checkTupleIdentical(self._size, _tally._size)
            arr_new = self._arr * _tally._arr
        return Tally(arr_new, self._origin, self._size)
    
    def __truediv__(self, _tally):
        if type(_tally) != Tally:
            raise TypeError()
        checkTupleIdentical(self._origin, _tally._origin)
        checkTupleIdentical(self._size, _tally._size)

        arr_new = self._arr / _tally._arr
        return Tally(arr_new, self._origin, self._size)
