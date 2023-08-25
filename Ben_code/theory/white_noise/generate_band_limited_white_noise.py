# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:45:19 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import pdf_routines as pdf
import lmfit

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    print(len(idx))
    plt.plot(freqs,f)
    plt.show()
    return fftnoise(f)

min_freq = 1.0
max_freq = 1000.0
samples = 2**20
samplerate = 1.0E4

time_series = band_limited_noise(min_freq,max_freq,samples,samplerate)
time_axis = np.linspace(0,len(time_series)/samplerate,len(time_series))
plt.plot(time_axis,time_series)
plt.show()

offset = 1000
window = 1000
window_samples = time_series[offset:offset+window]
window_times = time_axis[offset:offset+window]
cutoff_freq = 1/window_times[-1]
print('cutoff freq = '+str(cutoff_freq))
plt.plot(window_times,window_samples)
plt.show()

n_bins = 20
hist,bins = pdf.get_hist(window_samples,n_bins)


model = lmfit.Model(pdf.gaussian_pdf)
params = model.make_params()
params['m'].value = 0.0
params['s'].value = 0.0002

result = model.fit(hist,params=params,x=bins)
print(result.fit_report())

plt.plot(bins,hist,'.')
plt.plot(bins,result.best_fit)
plt.show()

