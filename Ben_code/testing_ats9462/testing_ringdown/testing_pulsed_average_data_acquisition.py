# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:31:57 2019

@author: Ben
"""

import instrument_classes_module as icm
import numpy as np
import math
import matplotlib.pyplot as plt
import lmfit

record_length = 200.0E-6# record length in seconds
n_records = 10000

samplerate = 180000000.0
n_samples = math.ceil(record_length*samplerate)

myboard = icm.ats9462(input_range=0.2,BW_limit = 0,clock_source='int')
memorySize_samples, bitsPerSample = myboard.board.getChannelInfo()
bytesPerSample = (bitsPerSample.value + 7) // 8

time_axis,voltage_data_A = myboard.acquire_NPT_average(record_length = record_length, \
                                                                      n_records = n_records, \
                                                                     channel_str = 'A')






if_freq = 5.0E6

iwave = np.cos(2*np.pi*if_freq*time_axis)
qwave = np.sin(2*np.pi*if_freq*time_axis)

samples = len(time_axis)
samples_per_period = int(round(samples/((time_axis[-1]-time_axis[0])*if_freq)))

iprod = iwave*voltage_data_A
qprod = qwave*voltage_data_A
inphase = 2*np.convolve(iprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
quadrature = 2*np.convolve(qprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
amplitude_vs_t = np.sqrt((inphase**2)+(quadrature**2))
phase_vs_t = np.arctan(quadrature/inphase)


time_axis = time_axis/1.0E-6


plt.plot(time_axis,voltage_data_A)
plt.show()

plt.plot(time_axis[100:200]/1.0E-6,voltage_data_A[0:100])
plt.xlabel('Time (us)')
plt.title('second 100 points')
plt.show()

plt.plot(time_axis[-100:],voltage_data_A[-100:])
plt.title('last 100 points')
plt.show()

max_ind = len(inphase)
plt.plot(time_axis[0:max_ind],inphase)
plt.title('in phase')
plt.show()

plt.plot(time_axis[0:max_ind],quadrature)
plt.title('quadrature')
plt.show()

plt.plot(time_axis[0:max_ind],amplitude_vs_t)
plt.title('amplitude')
plt.show()

plt.plot(time_axis[0:max_ind],phase_vs_t)
plt.title('phase')
plt.show()


decay_ind_low = 1700
decay_ind_high = 1820

plt.plot(time_axis[decay_ind_low:decay_ind_high],amplitude_vs_t[decay_ind_low:decay_ind_high])
plt.title('amplitude decay')
plt.show()

plt.plot(time_axis[decay_ind_low:decay_ind_high],voltage_data_A[decay_ind_low:decay_ind_high])
plt.title('signal decay')
plt.show()

def exp_decay(time,rate,initial,dc_offset):
    result = (initial*np.exp(-1*rate*(time-time[0])))+dc_offset
    return result

model = lmfit.Model(exp_decay)
params = model.make_params()
params['rate'].value = 1.0E6
params['initial'].value = 0.001
params['dc_offset'].value=0.0
params['dc_offset'].vary=False

time_window = time_axis[decay_ind_low:decay_ind_high]-time_axis[decay_ind_low]
amplitude_window = amplitude_vs_t[decay_ind_low:decay_ind_high]

fit_result = model.fit(amplitude_window,\
                       params=params,\
                       time=time_window)

print(fit_result.fit_report())

amplitude_fit = fit_result.best_fit

plt.plot(time_window,amplitude_window,'.')
plt.plot(time_window,amplitude_fit)
plt.title('amplitude decay with fit')
plt.show()

print('kappa_tot/2pi = '+str((fit_result.params['rate'].value/np.pi)/1.0E6)+' MHz')