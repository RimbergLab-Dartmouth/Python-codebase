# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:06:56 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import pdf_routines as pdf

def s11(delta,xi):
    numerator = delta-(0.5j*((1-xi)/(1+xi)))
    denominator = delta-0.5j
    return numerator/denominator

n_samples = 100000

xi=3.5
sigma = 0.7
delta0 = 0.0
delta_samples = np.random.normal(delta0,sigma,size=n_samples)
s11_samples = s11(delta_samples,xi)

noise_sigma = 0.15
x_noise_samples = np.random.normal(0.0,noise_sigma,size=n_samples)
y_noise_samples = np.random.normal(0.0,noise_sigma,size=n_samples)

real_samples = np.real(s11_samples)+x_noise_samples
imag_samples = np.imag(s11_samples)+y_noise_samples

n_bins = 30
hist2d, real_edges, imag_edges = np.histogram2d(real_samples,
                                                imag_samples,
                                                bins=n_bins,
                                                density=True)

scale = 7
fig, ax = plt.subplots(figsize=(scale,scale))
mesh = ax.pcolormesh(real_edges, imag_edges, hist2d.T, cmap='inferno')
ax.set_title('S11 Histogram')
fig.colorbar(mesh, ax=ax)
ax.set_xlabel('Re(S11)')
ax.set_ylabel('Im(S11)')
ax.set_aspect('equal')
plt.show()


n_bins = 30
hist2d, real_edges, imag_edges = np.histogram2d(x_noise_samples,
                                                y_noise_samples,
                                                bins=n_bins,
                                                density=True)

scale = 7
fig, ax = plt.subplots(figsize=(scale,scale))
mesh = ax.pcolormesh(real_edges, imag_edges, hist2d.T, cmap='inferno')
ax.set_title('S11 Noise Hist')
fig.colorbar(mesh, ax=ax)
ax.set_xlabel('Re(S11)')
ax.set_ylabel('Im(S11)')
ax.set_aspect('equal')
plt.show()
