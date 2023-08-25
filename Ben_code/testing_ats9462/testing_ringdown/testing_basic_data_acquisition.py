# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:28:22 2019

@author: Ben
"""

import instrument_classes_module as icm
import ats_module as atsm
import numpy as np
import time
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import struct


awg = icm.tektronix_awg520(10)
awg.set_timeout(60000)
awg.set_frequency_reference(1,'ext')
awg.set_run_mode('trig')
awg.toggle_output(1,1)
awg.set_markers_low(1,-1)
awg.set_markers_high(1,1)

awg_sample_rate = 1.0E9
waveform_duration = 1.0E-6
waveform_delay1 = 4.0E-6
waveform_delay2 = 4.0E-6
total_duration = waveform_duration + waveform_delay1 + waveform_delay2
waveform_samples = math.ceil(waveform_duration*awg_sample_rate)
delay_samples1 = math.ceil(waveform_delay1*awg_sample_rate)
delay_samples2 = math.ceil(waveform_delay2*awg_sample_rate)
total_samples = delay_samples1+waveform_samples+delay_samples2

waveform_time_axis = np.linspace(0,waveform_duration,waveform_samples)
if_freq = 20.0E6
if_amplitude = 0.3
if_phase = 0
waveform = np.zeros(total_samples)
waveform[delay_samples1:total_samples-delay_samples2] = if_amplitude*np.cos((if_freq*waveform_time_axis)+if_phase)
m1 = np.ones(total_samples,dtype=int)
m1[0] = 0
m1[-1] = 0
m2 = np.zeros(total_samples,dtype=int)

wfm_filename = 'test.wfm'
awg_dir = 'testing_oct_2019'
awg.reset_cwd()
awg.mkdir(awg_dir)
awg.set_cwd(awg_dir)
awg.send_waveform(waveform,m1,m2,wfm_filename,awg_sample_rate)
awg.load_waveform(1,wfm_filename)
awg.run()

print('Waveform Loaded')

filename = 'test_data.bin'
record_length = total_duration# record length in seconds
records_per_buffer = 1
total_buffers = 1

samplerate = 180000000.0
n_samples = math.ceil(record_length*samplerate)

myboard = icm.ats9462(input_range=0.8,clock_source='int')
memorySize_samples, bitsPerSample = myboard.board.getChannelInfo()
bytesPerSample = (bitsPerSample.value + 7) // 8

time_axis,voltage_data_A,voltage_data_B = myboard.acquire_NPT_single(record_length = record_length, \
                                                                     triggering_instrument = awg, \
                                                                     channel_str = 'AB')


plt.plot(time_axis,voltage_data_A)
plt.show()
plt.plot(time_axis,voltage_data_B)
plt.show()