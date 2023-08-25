# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:13:07 2018

@author: Ben
"""

import numpy as np
import instrument_classes_module as icm

awg = icm.tektronix_awg520(8)

samplerate = 10.**9 # 1 GS/s
total_duration = 10.**-6 # one microsecond duration
pulse_duration = 10.**-7 # 100ns pulse
delay =  10.**-8 #start the pulse at the 10ns mark

n_samples = math.ceil(samplerate*total_duration)
waveform = np.zeros(n_samples)
m1 = np.zeros(n_samples) # marker 1 rises at start of pulse
m2 = np.zeros(n_samples)