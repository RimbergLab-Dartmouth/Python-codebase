# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:55:17 2019

@author: Ben
"""


import numpy as np
import matplotlib.pyplot as plt
import lmfit

if_freq = 10.0*(10**6)
sample_rate = 180.0*(10**6)
tscale = 1.0/sample_rate

n_cycles = 100
samples = int(round(sample_rate*(1.0/if_freq)*n_cycles))
duration = tscale*samples
time_axis = np.linspace(0,duration,num=samples)

initial_amplitude = 0.7345
decay_rate = 1.0E6
initial_phase = 0.8345
snr = 2.0
rms_noise_amplitude = initial_amplitude/snr

noise_signal = np.random.normal(0.0,rms_noise_amplitude,samples)
if_signal = initial_amplitude*np.cos((2*np.pi*if_freq*time_axis)-initial_phase)*np.exp(-1*decay_rate*time_axis)
if_signal += noise_signal
samples_per_period = round(0.5*(sample_rate/if_freq))

iwave = np.cos(2*np.pi*if_freq*time_axis)
qwave = np.sin(2*np.pi*if_freq*time_axis)

iprod = iwave*if_signal
qprod = qwave*if_signal

inphase = 2*np.convolve(iprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
quadrature = 2*np.convolve(qprod,np.ones(samples_per_period)/samples_per_period,mode='valid')
amplitude = np.sqrt((inphase**2)+(quadrature**2))
phase = np.arctan(quadrature/inphase)
print('length before = '+str(len(iprod)))
print('length after = '+str(len(inphase)))
print('samples per period = '+str(samples_per_period))

max_ind = len(quadrature)
averaged_time_axis = time_axis[0:max_ind]
average_inphase = sum(inphase)/len(inphase)
print('average i: '+str(average_inphase))
average_quadrature = sum(quadrature)/len(quadrature)
print('average q: '+str(average_quadrature))
measured_phase = np.arctan(average_quadrature/average_inphase)
measured_amplitude = np.sqrt((average_inphase**2)+(average_quadrature**2))
print('actual phase: '+str(initial_phase))
print('measured phase: '+str(measured_phase))
print('actual amplitude '+str(initial_amplitude))
print('measured avg amplitude: '+str(measured_amplitude))
print('measured initial amplitude: '+str(amplitude[0]))

plt.plot(time_axis,if_signal)
plt.xlabel('time')
plt.ylabel('f(t)')
plt.title('raw signal vs t')
plt.show()

plt.plot(averaged_time_axis,phase)
plt.xlabel('time')
plt.ylabel('phase')
plt.title('phase vs t')
plt.show()

plt.plot(averaged_time_axis, amplitude)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('amplitude vs t')
plt.show()

def exp_decay(time,rate,initial):
    result = initial*np.exp(-1*rate*(time-time[0]))
    return result

model = lmfit.Model(exp_decay)
params = model.make_params()
params['rate'].value = decay_rate*2.0
params['initial'].value = initial_amplitude

fit_result = model.fit(amplitude,\
                       params=params,\
                       time=averaged_time_axis)

print(fit_result.fit_report())
print('actual decay rate = '+str(decay_rate))
print('fit decay rate = '+str(fit_result.params['rate'].value))

