# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:54:13 2018

@author: Ben
"""

import numpy as np
import math
import struct
import visa
import gpib_module as gpib
import instrument_classes_module as icm


awg = icm.tektronix_awg520(8)

filename = 'test_waveform5.wfm'

samplerate = 10.**9 # 1 GS/s
total_duration = 10.**-6 # one microsecond duration
pulse_duration = 10.**-7 # 100ns pulse
delay =  10.**-8 #start the pulse at the 10ns mark

n_samples = math.ceil(samplerate*total_duration)
waveform = np.zeros(n_samples)
m1 = np.zeros(n_samples) # marker 1 rises at start of pulse
m2 = np.zeros(n_samples)

start_ind = math.ceil(delay*samplerate)
finish_ind = math.ceil(pulse_duration*samplerate)+start_ind

waveform[start_ind:finish_ind+1] = 1
m1[start_ind:] = 1 #let marker 1 mark the beginning of the pulse
m2[finish_ind:] = 1 #and marker 2 the end of the pulse

awg.send_waveform(waveform,m1,m2,filename,samplerate)
