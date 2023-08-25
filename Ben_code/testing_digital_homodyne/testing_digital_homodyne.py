# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:29:50 2018

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

if_freq = 0.5*(10**9)
sample_rate = 200.0*(10**9)
tscale = 1.0/sample_rate

samples = 10000
duration = tscale*samples
time_axis = np.linspace(0,duration,num=samples)

amplitude = 0.7345
phase = 0.8345
if_signal = amplitude*np.cos((2*np.pi*if_freq*time_axis)-phase)
samples_per_period = round(0.5*(sample_rate/if_freq))

iwave = np.cos(2*np.pi*if_freq*time_axis)
qwave = np.sin(2*np.pi*if_freq*time_axis)

iprod = iwave*if_signal
qprod = qwave*if_signal

inphase = 2*np.convolve(iprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
quadrature = 2*np.convolve(qprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
amplitude_vs_t = np.sqrt((inphase**2)+(quadrature**2))
phase_vs_t = np.arctan(quadrature/inphase)
print('length before = '+str(len(iprod)))
print('length after = '+str(len(inphase)))
print('samples per period = '+str(samples_per_period))

averaged_time_axis = time_axis[0:samples-samples_per_period-1]
print(len(averaged_time_axis))
print(len(quadrature))
average_inphase = sum(inphase)/len(inphase)
print('average i: '+str(average_inphase))
average_quadrature = sum(quadrature)/len(quadrature)
print('average q: '+str(average_quadrature))
measured_phase = np.arctan(average_quadrature/average_inphase)
measured_amplitude = np.sqrt((average_inphase**2)+(average_quadrature**2))
print('actual phase: '+str(phase))
print('measured phase: '+str(measured_phase))
print('actual amplitude '+str(amplitude))
print('measured amplitude: '+str(measured_amplitude))

plt.figure()
plt.plot(phase_vs_t)
plt.figure()
plt.plot(amplitude_vs_t)
plt.show()

plt.figure()
plt.plot(inphase)
plt.figure()
plt.plot(quadrature)
plt.show()