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
awg.set_timeout(60000)

waveform_name = 'freq_sweep'
sample_rate = 5.0*(10**9)

first_frequency = 350.0*(10**6)
last_frequency = 450.0*(10**6)
step_frequency = 1.0*(10**6)

waveform_duration = 1.0*(10**-6)
waveform_samples = round(sample_rate*waveform_duration)
pulse_duration = 500.0*(10**-9)
pulse_samples = round(sample_rate*pulse_duration)
delay_samples = waveform_samples-pulse_samples
print('waveform samples: '+str(waveform_samples))
print('pulse samples: '+str(pulse_samples))
print('delay samples: '+str(delay_samples))
first_ind = delay_samples//2
last_ind = first_ind+pulse_samples
print('first ind: '+str(first_ind))
print('last ind: '+str(last_ind))

n_frequencies = math.ceil((last_frequency-first_frequency)/step_frequency)
print('number of frequencies: '+str(n_frequencies))
total_samples = waveform_samples*n_frequencies
print('total samples: '+str(total_samples))

time_axis = np.linspace(0,pulse_duration,pulse_samples)
freq_axis = np.linspace(first_frequency,last_frequency,n_frequencies)

marker_delay = 17.0*(10**-9)
marker_delay_samples = round(sample_rate*marker_delay)
print('marker delay: '+str(marker_delay_samples))

waveform = np.zeros(total_samples)
m1 = np.zeros_like(waveform)
m2 = np.zeros_like(waveform)
waveform_ind = 0
for ii in range(n_frequencies):
    waveform[first_ind:last_ind]=np.sin(2*np.pi*freq_axis[ii]*time_axis)
    m1[first_ind+marker_delay_samples:last_ind+marker_delay_samples] = 1
    m2[waveform_ind:first_ind] = 1
    first_ind += waveform_samples
    last_ind += waveform_samples
    waveform_ind += waveform_samples
    
awg.send_waveform(waveform_name, waveform, m1, m2)
awg.load_waveform(2,waveform_name)
