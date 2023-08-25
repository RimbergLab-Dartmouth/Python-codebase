# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:56:56 2019

@author: Ben
"""


import instrument_classes_module as icm
import numpy as np
import math
import matplotlib.pyplot as plt

acquisition_length = 1.0# record length in seconds

sample_rate = 10.0E6

myboard = icm.ats9462(input_range=0.2,BW_limit = 0,clock_source='int',sample_rate=sample_rate)
memorySize_samples, bitsPerSample = myboard.board.getChannelInfo()
bytesPerSample = (bitsPerSample.value + 7) // 8

time_axis,voltage_data_A = myboard.acquire_TS_single(acquisition_length = acquisition_length, \
                                                                     channel_str = 'A')

plt.plot(time_axis,voltage_data_A)
plt.show()

plt.plot(time_axis[100:200]/1.0E-6,voltage_data_A[0:100])
plt.xlabel('Time (us)')
plt.title('second 100 points')
plt.show()

plt.plot(time_axis[-100:],voltage_data_A[-100:])
plt.title('last 100 points')
plt.show()

