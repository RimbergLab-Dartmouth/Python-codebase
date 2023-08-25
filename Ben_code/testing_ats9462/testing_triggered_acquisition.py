# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:54:13 2018

@author: Ben
"""

import numpy as np
import math
import struct
import visa
import instrument_classes_module as icm

awg = icm.tektronix_awg520(8)
awg.reset_cwd()
awg.cd('testdir/testing_adc')
filename = 'test_pulse.wfm'

samplerate = 10.**9 # 1 GS/s
total_duration = 10.**-6 # one microsecond duration
pulse_duration = 3*(10.**-7) # 300ns pulse
delay =  10.**-7 #start the pulse at the 100ns mark

n_samples = math.ceil(samplerate*total_duration)
waveform = np.zeros(n_samples)
m1 = np.zeros(n_samples)
m2 = np.zeros(n_samples)

start_ind = math.ceil(delay*samplerate)
finish_ind = math.ceil(pulse_duration*samplerate)+start_ind

waveform[start_ind:finish_ind+1] = 1
waveform = waveform*2 - 1
m1[1:] = 1 #let marker 1 mark t = 0
m2[start_ind:] = 1 #and marker 2 the start of the pulse

print(awg.query_cwd())

awg.send_waveform(waveform,m1,m2,filename,samplerate)
