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
awg.set_timeout(30000)

n_pulses = 100 # number of pulses, and therefore number of phase points
total_phase = (2*np.pi)
delta_phi = total_phase/n_pulses
frequency = 45.0*(10**6) # 45MHz carrier
amplitude = 1.0

awg_sample_rate = 10.**9
waveform_duration = 10.**-6
pulse_duration = 3.0*(10**-7) # 300 ns pulse
delay =  1.0*(10.**-8) # start the first pulse at the 10 ns mark

samples_per_waveform = math.ceil(awg_sample_rate*waveform_duration)
total_samples = n_pulses*samples_per_waveform

start_pulse_ind = math.ceil(delay*awg_sample_rate)
finish_pulse_ind = math.ceil(pulse_duration*awg_sample_rate)+start_pulse_ind
samples_per_pulse = finish_pulse_ind - start_pulse_ind + 1

waveform = np.zeros(total_samples)
m1 = np.zeros(total_samples) # marker 1 indicates the start of the waveform
m1[1:] = 1
m2 = np.zeros(total_samples) # marker 2 does nothing

sample_time_axis = np.linspace(0,pulse_duration,samples_per_pulse)
phase_array = 2*np.pi*frequency*sample_time_axis
phase = 0
for ii in range(n_pulses):
    waveform[start_pulse_ind:start_pulse_ind + samples_per_pulse] = \
            amplitude*np.sin(phase_array + phase)
    
    phase += delta_phi
    start_pulse_ind += samples_per_waveform


awg.send_waveform(waveform,m1,m2,'phase_sweep.wfm',awg_sample_rate)


