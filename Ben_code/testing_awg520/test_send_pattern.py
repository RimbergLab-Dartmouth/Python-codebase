# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:29:08 2018

@author: Ben
"""

import numpy as np
import math
import visa
import gpib_control as gpib

awg = gpib.connect(8)

samplerate = 10.**9 # 1 GS/s
total_duration = 10.**-6 # one microsecond duration
pulse_duration = 10.**-7 # 100ns pulse
delay =  10.**-8 #start the pulse at the 10ns mark

n_samples = samplerate*total_duration
pattern = np.zeros(n_samples)
m1 = np.zeros(n_samples)
m2 = np.zeros(n_samples)

start_ind = math.ceil(delay*samplerate)
finish_ind = math.ceil(pulse_duration*samplerate)+start_ind

pattern[start_ind:finish_ind] = 1
m1[start_ind] = 1 #let marker 1 mark the rising edge
m2[finish_ind] = 1 #and marker 2 the falling edge