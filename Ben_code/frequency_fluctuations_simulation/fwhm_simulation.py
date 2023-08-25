# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:23:50 2019

@author: Ben
"""

import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

figs_dir = 'figs/'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

chi=1.0
n_sigmas = 50
sigma_start = 0.03
sigma_stop = 2.0
sigma_array = np.linspace(sigma_start,sigma_stop,n_sigmas)

fit_ind = int(round(n_sigmas/5))

tolerance = 1.0E-13
n_evals = int(round(1.0E6))


def s21_sqmag(df,chi,sigma):
    argument = (df-(1j/2.0))/(sigma*np.sqrt(2.0))
    factor = (-1.0/sigma)*(np.sqrt(chi)/(1+chi))
    term1 = np.sqrt(np.pi/2.0)*np.exp(-1.0*(argument**2))
    term2 = -1j*np.sqrt(2.0)*sp.special.dawsn(argument)
    gamma = factor*(term1+term2)
    gamma_sqmag = np.abs(gamma)**2
    return gamma_sqmag
def make_s21_fwhm(chi,sigma):
    val0 = s21_sqmag(0.0,chi,sigma)/2
    def s21_fwhm(detuning):
        result = s21_sqmag(detuning,chi,sigma) - val0
        return result
    return s21_fwhm



fwhm_array = np.zeros(n_sigmas)
for ii in range(n_sigmas):
    this_sigma = sigma_array[ii]
    detuning1 = opt.fsolve(make_s21_fwhm(chi,this_sigma),1/2,\
                                   xtol=tolerance,\
                                   maxfev=n_evals,\
                                   factor=0.001)
    #detuning1 = sol1.x
    detuning2 = opt.fsolve(make_s21_fwhm(chi,this_sigma),-1/2,\
                                   xtol=tolerance,\
                                   maxfev=n_evals,\
                                   factor=0.001)
    #detuning2 = sol2.x
    fwhm_array[ii] = np.abs(detuning1-detuning2)
    
plt.plot(sigma_array,fwhm_array,'.')
plt.show()


coeffs = np.polyfit(sigma_array[fit_ind:],fwhm_array[fit_ind:],1)
print('slope = '+str(coeffs[0]))
print('intercept = '+str(coeffs[1]))
fwhm_fit = (coeffs[0]*sigma_array)+coeffs[1]

scale = 12
ticklabelsize = 20
labelsize = 24
plotlabelsize = 20


plt.figure(figsize=(scale,scale))
plt.plot(sigma_array,fwhm_array,'.')
plt.plot(sigma_array,fwhm_fit)
plt.xlabel('$\sigma/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
plt.ylabel('$\mathrm{FWHM}/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = ticklabelsize)
ax.tick_params(axis = 'both', which = 'minor', labelsize = ticklabelsize)
plt.savefig(figs_dir+'fwhm_simulation.png')
plt.show()
