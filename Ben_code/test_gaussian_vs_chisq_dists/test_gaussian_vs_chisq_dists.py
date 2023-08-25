# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:56:11 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import pdf_routines as pdf

s = 1.5

n_samples = 10000
samples = np.random.normal(0.0,s,n_samples)

transformed = (1+samples)**2
gaussian_part = 1+(2*samples)
chisq_part = 1+(samples**2)

n_bins = 40
hist,bins = pdf.get_hist(transformed,n_bins,density=True)
gaussian_hist,gaussian_bins = pdf.get_hist(gaussian_part,n_bins,density=True)
chisq_hist,chisq_bins = pdf.get_hist(chisq_part,n_bins,density=True)

plt.plot(bins,hist,'.')
plt.plot(gaussian_bins,gaussian_hist,'.')
plt.plot(chisq_bins,chisq_hist,'.')
plt.xlabel('bins')
plt.ylabel('pdf')
plt.title('Full, Gaussian Part, and Chisq part')
plt.show()

plt.plot(bins,hist,'.')
plt.plot(chisq_bins,chisq_hist,'.')
plt.xlabel('bins')
plt.ylabel('pdf')
plt.title('Full and Chisq part')
plt.show()

plt.plot(bins,hist,'.')
plt.plot(gaussian_bins,gaussian_hist,'.')
plt.xlabel('bins')
plt.ylabel('pdf')
plt.title('Full and Gaussian Part')
plt.show()




