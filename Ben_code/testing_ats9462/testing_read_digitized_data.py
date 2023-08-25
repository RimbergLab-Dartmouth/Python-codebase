# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:32:10 2018

@author: Ben
"""

import ctypes
import numpy as np
import os
import signal
import sys
import time
import atsapi as ats
import math
import binascii
import matplotlib
import matplotlib.pyplot as plt
import struct
import ats_module as atsm

filename = 'test_data.bin'

samplerate = 180000000.0

record_length = 2.0*(10**-6) # record length in seconds
n_samples = math.ceil(record_length*samplerate)

board = ats.Board(systemId = 1, boardId = 1)
memorySize_samples, bitsPerSample = board.getChannelInfo()
bytesPerSample = (bitsPerSample.value + 7) // 8
#print(bytesPerSample)


myboard = atsm.ats9462()
data_list = myboard.acquire_NPT(record_length = record_length, save_data = True,filename='test_data.bin',return_data=True)

  
#print(type(data_list))
#print(data_list[0])
data = []

f = open(filename,'rb')
for ii in range(n_samples):
    chunk = f.read(bytesPerSample)
    decoded_chunk = struct.unpack('H',chunk)
    data.append(decoded_chunk)

#rest = f.read()
#print(rest)
f.close()
#print('\n')
#print(data)
#print(type(data[1]))



max_voltage = 0.2
#for ii in range(len(data)):
    
plt.figure()
plt.plot(data[0:100],marker='.',linestyle='None')
plt.figure()
plt.plot(data,marker='.',linestyle='None')

plt.show()

#print('start of data:')
#print(data[0:100])
#print('\n')

#print('all data:')
#print(data)
#print('\n')

"""
print(len(data))
print(data)

decoded_data = struct.unpack('>e',data)
print(decoded_data)

data = f.read(bytesPerSample)
print(len(data))
print(data)

decoded_data = struct.unpack('>e',data)
print(decoded_data)

data = f.read(bytesPerSample)
print(len(data))
print(data)

decoded_data = struct.unpack('>e',data)
print(decoded_data)

data = f.read(bytesPerSample)
print(len(data))
print(data)

decoded_data = struct.unpack('>e',data)
print(decoded_data)
"""