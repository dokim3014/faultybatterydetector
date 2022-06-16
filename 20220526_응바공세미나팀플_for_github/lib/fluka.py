import numpy as np
import matplotlib.pyplot as plt
import struct


class USRBIN():
    def __init__(self, fileName):
        file = open(fileName, "rb")
        self.line = file.read()
        file.close()
        self.pointer = 0
    def Reset(self):
        self.pointer = 0
    def Get(self):
        length = struct.unpack("i",self.line[self.pointer:self.pointer+4])[0]
        if self.line[self.pointer:self.pointer+4] != self.line[self.pointer+4+length:self.pointer+8+length]:
            print("Wrong binary file structure")
            return
        self.pointer += 4
        output = self.line[self.pointer:self.pointer+length]
        self.pointer += length + 4
        return output
    
"""
fluka = USRBIN("Epithermal_50.bnn")

_ = fluka.Get()
_ = fluka.Get()

dataByte = fluka.Get()
arr1D = np.frombuffer(dataByte, dtype=np.float32)
arr3D = arr1D.reshape((50,50,50))
arr3D = np.transpose(arr3D, axes=(2,1,0))
"""
