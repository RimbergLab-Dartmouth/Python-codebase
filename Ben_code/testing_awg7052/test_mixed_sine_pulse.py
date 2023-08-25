# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:29:08 2018

@author: Ben
"""

import numpy as np
import math
import struct
import visa
import instrument_classes_module as icm


awg = icm.tektronix_awg7052(1)

waveform_name = 'sine_pulse'
sample_rate = 5.0*(10**9)
frequency = 400.0*(10**6)
period = 1.0/frequency
n_periods = 20
duration = period*n_periods
sine_samples = math.ceil(n_periods*(sample_rate/frequency))
extra_samples_before = 20
extra_samples_after = 20
extra_samples = extra_samples_before + extra_samples_after
n_samples = sine_samples + extra_samples

time_axis = np.linspace(0,duration,sine_samples)
phase_axis = 2*np.pi*frequency*time_axis
sine_wave = np.sin(phase_axis)

waveform = np.zeros(n_samples)
waveform[extra_samples_before:extra_samples_before+sine_samples] = sine_wave

m1 = np.zeros(n_samples) # marker 1 rises at start of pulse
m2 = np.zeros(n_samples)

m1_start = math.ceil(n_samples/4)
m1_end = math.ceil(3*m1_start)
m1[m1_start:m1_end] = 1

m2_start = math.ceil(n_samples/2)
m2[m2_start:] = 1

waveform[-1]=0

awg.send_waveform(waveform_name, waveform, m1, m2)
awg.load_waveform(2,waveform_name)