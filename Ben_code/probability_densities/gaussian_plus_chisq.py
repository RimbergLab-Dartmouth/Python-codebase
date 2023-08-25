# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:53:43 2020

@author: Ben
"""

import numpy as np
import pdf_routines as pdf
import matplotlib.pyplot as plt
import scipy as sp

n_samples = 100000
n_dofs = 2
kerr_scale = 4.0
chisq_mean = 0.0

chisq_scale = kerr_scale/4.0

gaussian_samples = np.random.normal(size=n_samples)
chisq_samples = np.zeros(n_samples,dtype=float)
for ii in range(n_dofs):
    chisq_samples += np.random.normal(loc=chisq_mean,size=n_samples)**2
chisq_samples = chisq_samples*chisq_scale
samples = gaussian_samples+chisq_samples

renormalized_mean = 2*chisq_scale
renormalized_sigma = np.sqrt(1.0 + ((2*chisq_scale)**2))

n_bins = 30
hist,bins = pdf.get_hist(samples,n_bins)
fit = pdf.gaussian_pdf(bins,renormalized_mean,renormalized_sigma)

plt.plot(bins,hist,'.')
plt.plot(bins,fit)
plt.xlabel('bins')
plt.ylabel('hist')
plt.show()

def analytic_pdf(x,sigma,K):
    factor = np.exp((2*((sigma/K)**2))-((2*x)/K))
    result = factor*((1/np.abs(K))-(sp.special.erf(((np.sqrt(2)*sigma)/K)-(x/(np.sqrt(2)*sigma)))/K))
    return result

n_points = 1000
xlow = -5
xhigh = 10
x_axis = np.linspace(xlow,xhigh,n_points)
exact = analytic_pdf(x_axis,1.0,kerr_scale)
approximate = pdf.gaussian_pdf(x_axis,renormalized_mean,renormalized_sigma)

plt.plot(x_axis,exact)
plt.plot(x_axis,approximate)
plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.show()






