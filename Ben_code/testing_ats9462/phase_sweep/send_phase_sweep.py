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


awg = icm.tektronix_awg520(8)

n_waveforms = 50 # number of pulses, and therefore number of phase points

awg_sample_rate = 10.**9
waveform_duration = 10.**-6
pulse_duration = 3.0*(10**-7) # 300 ns pulse
delay =  0.1*(10.**-8) # start the first pulse at the 10 ns mark

samples_per_waveform = math.ceil(awg_sample_rate*waveform_duration)
total_samples = n_waveforms*samples_per_waveform

start_waveform_ind = 1
marker_up_duration = 500
start_pulse_ind = math.ceil(delay*awg_sample_rate)
finish_pulse_ind = math.ceil(pulse_duration*awg_sample_rate)+start_pulse_ind

waveform = np.zeros(total_samples)
m1 = np.zeros(total_samples) # marker 1 indicates the start of each waveform
m2 = np.zeros(total_samples) # marker 2 does nothing

for ii in range(n_waveforms):

    waveform[start_pulse_ind:finish_pulse_ind+1] = 1
    m1[start_waveform_ind:start_waveform_ind+marker_up_duration] = 1 #let marker 1 mark the beginning of the pulse
    
    start_waveform_ind += samples_per_waveform
    start_pulse_ind += samples_per_waveform + 1
    finish_pulse_ind += samples_per_waveform + 1

waveform = waveform*2 - 1
awg.send_waveform(waveform,m1,m2,'phase_sweep.wfm',awg_sample_rate)


