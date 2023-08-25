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
n_samples = 20

time_axis = np.linspace(0,2*np.pi,n_samples)
waveform = np.sin(time_axis)

m1 = np.zeros(n_samples) # marker 1 rises at start of pulse
m2 = np.zeros(n_samples)

m1_start = math.ceil(n_samples/4)
m1_end = math.ceil(3*m1_start)
m1[m1_start:m1_end] = 1

m2_start = math.ceil(n_samples/2)
m2[m2_start:] = 1

waveform = waveform

awg.send_waveform(waveform_name, waveform, m1, m2)