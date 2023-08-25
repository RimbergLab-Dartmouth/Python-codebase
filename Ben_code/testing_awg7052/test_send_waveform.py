# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:29:08 2018

@author: Ben
"""

import numpy as np
import math
import struct
import visa
import gpib_module as gpib
import instrument_classes_module as icm


awg = icm.tektronix_awg7052(1)

waveform_name = 'test_waveform'
n_samples = 1000
awg.new_waveform(waveform_name,n_samples)

time_axis = np.linspace(0,2*np.pi,n_samples)
waveform = np.sin(time_axis)


m1 = np.zeros(n_samples) # marker 1 rises at start of pulse
m2 = np.zeros(n_samples)

m1_start = math.ceil(n_samples/4)
m1_end = math.ceil(3*m1_start)
m1[m1_start:m1_end] = 1

m2_start = math.ceil(n_samples/2)
m2[m2_start:] = 1

m = ((2**7)*m2) + ((2**6)*m1)

bytes_data = b''
for ii in range(n_samples):
    bytes_data += struct.pack('fB',waveform[ii],int(m[ii]))

num_bytes = n_samples*5
num_bytes = str(num_bytes)
num_digits = str(len(num_bytes))

num_bytes = num_bytes.encode('ascii')
num_digits = num_digits.encode('ascii')
bytes_count = num_digits + num_bytes

bytes_name = waveform_name.encode('ascii')
bytes_samples = str(n_samples)
bytes_samples = bytes_samples.encode('ascii')
message = b'wlis:wav:data "'+bytes_name+b'",0,'+bytes_samples+b',#'+bytes_count
message += bytes_data

awg.write_raw(message)