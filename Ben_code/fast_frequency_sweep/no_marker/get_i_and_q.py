# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:50:31 2018

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt



f = open('voltage.dat','r')
voltage = f.read()
f.close()

voltage_list = voltage.split(',')
voltage_list = [float(item) for item in voltage_list]

f = open('time.dat')
time = f.read()
f.close()

time_list = time.split(',')
time_list = [float(item) for item in time_list]
tscale = time_list[1]-time_list[0]

sample_rate = 20*(10**9)
pulse_time = 500.0*(10**-9)
samples_per_pulse = round(sample_rate*pulse_time)
waveform_time = 1.0*(10**-6)
samples_per_waveform = round(sample_rate*waveform_time)

n_frequencies = 100
lo_freq = 2.0*(10**9)
if_freq = 350.0*(10**6)
if_freq_inc = 1.0*(10**6)
freq_list = []

start_ind = (53*20000)+14408
pulse_list = []

time_axis = np.linspace(0,pulse_time,samples_per_pulse)

inphase_list = []
quadrature_list = []
average_inphase_list = []
average_quadrature_list = []
average_phase_list = []
average_amplitude_list = []

for ii in range(n_frequencies):
    pulse = np.array(voltage_list[start_ind:start_ind+samples_per_pulse])
    pulse_list.append(pulse)
    
    freq = lo_freq - if_freq
    freq_list.append(freq)
    
    iwave = np.cos(2*np.pi*freq*time_axis)
    iprod = iwave*pulse
    qwave = np.sin(2*np.pi*freq*time_axis)
    qprod = qwave*pulse
    
    samples_per_period = round(1.0/(freq*tscale))
    
    inphase = []
    quadrature = [] 
    for jj in range(samples_per_pulse-samples_per_period-1):
        inphase.append(np.sum(iprod[jj:jj+samples_per_period])/(0.5*samples_per_period))
        quadrature.append(np.sum(qprod[jj:jj+samples_per_period])/(0.5*samples_per_period))
        
    inphase_list.append(inphase)
    quadrature_list.append(quadrature)
    
    avg_i = sum(inphase)/len(inphase)
    average_inphase_list.append(avg_i)
    avg_q = sum(quadrature)/len(quadrature)
    average_quadrature_list.append(avg_q)
    avg_phase = (np.arctan((avg_q/avg_i))-0.1)%np.pi
    average_phase_list.append(avg_phase)
    avg_amplitude = np.sqrt((avg_i**2)+(avg_q**2))
    average_amplitude_list.append(avg_amplitude)
    
    start_ind += samples_per_waveform
    if_freq += if_freq_inc
    
    #if ii%5==0:
        #plt.figure()
        #plt.plot(pulse[100:200])
        #plt.plot(inphase)
        #plt.figure()
        #plt.plot(quadrature)
        #if ii==10:
        #    plt.figure()
        #    plt.plot(iwave)
        #    plt.figure()
        #    plt.plot(iprod)
        
plt.figure()
plt.plot(freq_list,average_phase_list)
plt.figure()
plt.plot(freq_list,average_amplitude_list)
plt.show()
    

