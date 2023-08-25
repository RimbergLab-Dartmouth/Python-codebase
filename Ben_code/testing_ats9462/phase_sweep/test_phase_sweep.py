# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:14:41 2018

@author: Ben
"""

import instrument_classes_module as icm
import ats_module as atsm
import numpy as np
import time
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import struct

filename = 'test_data.bin'
record_length = 100.0*(10**-6) # record length in seconds
records_per_buffer = 1
total_buffers = 1

samplerate = 180000000.0
n_samples = math.ceil(record_length*samplerate)

myboard = atsm.ats9462()
memorySize_samples, bitsPerSample = myboard.board.getChannelInfo()
bytesPerSample = (bitsPerSample.value + 7) // 8


data_list = myboard.acquire_NPT(record_length = record_length, \
                                recordsPerBuffer = records_per_buffer, \
                                buffersPerAcquisition = total_buffers, \
                                buffer_count = total_buffers, \
                                save_data = True, \
                                filename=filename, \
                                return_data=False)
"""
for ii in range(len(data_list)):
    plt.figure()
    plt.plot(data_list[ii],marker='.',linestyle='None')
"""

data = []
f = open(filename,'rb')
for ii in range(n_samples*records_per_buffer*total_buffers):
    chunk = f.read(bytesPerSample)
    decoded_chunk = struct.unpack('H',chunk)
    data.append(decoded_chunk)
#rest = f.read()
#print(rest)
f.close()

plt.figure()
plt.plot(data,marker=',',linestyle='None')

plt.show()