# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:23:50 2019

@author: Ben
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpmath

decimal_precision = 50
mpmath.mp.dps = decimal_precision

chi=1.0
n_sigmas = 20
sigma_start = 0.0111
sigma_stop = 2.0
sigma_array = np.linspace(sigma_start,sigma_stop,n_sigmas)

fit_ind = int(round(n_sigmas/5))

tolerance = 1.0E-13
n_evals = int(round(1.0E6))

"""
def make_s21_sqmag(chi,sigma):
    g_ext = chi/(1+chi)
    def s21_sqmag(df):
        term = (df-(1j/2))/(sigma*np.sqrt(2))
        #term = term.astype(np.complex128)
        term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))
        term2 = 1j*np.exp(-1*(term**2))
        term3 = (2/np.sqrt(np.pi))*sp.special.dawsn(term)
        gamma = (term1*(term2+term3))
        gamma_sqmag = np.abs(gamma)**2
        return gamma_sqmag
    return s21_sqmag

def make_s21_fwhm(chi,sigma):
    s21_sqmag_fcn = make_s21_sqmag(chi,sigma)
    val0 = s21_sqmag_fcn(0)/2
    def s21_fwhm(df):
        result = s21_sqmag_fcn(df) - val0
        return result
    return s21_fwhm
"""
# NOTE THAT THESE FUNCTIONS EXPECT A SINGLE ARGUMENT DETUNING, NOT AN ARRAY
def s21_sqmag(df,chi,sigma):
    g_ext = chi/(1+chi)
    if isinstance(df,type(1.0)):
        df = mpmath.mpc(df)
    elif isinstance(df,type(np.array(1.0))):
        if len(df) != 1:
            print('ERROR: MORE THAN 1 ELEMENT IN df!')
        else:
            df = mpmath.mpc(df[0])
    term = (df-(1j/2.0))/(sigma*np.sqrt(2))
    """
    term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))
    term2 = 1j*np.exp(-1*(term**2))
    term3 = (2/np.sqrt(np.pi))*sp.special.dawsn(term)
    gamma = (term1*(term2+term3))
    """
    term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))*mpmath.exp(-1*(term**2))
    term2 = (1j+mpmath.erfi(term))
    gamma = term1*term2
    gamma_sqmag = mpmath.fabs(gamma)**2
    return float(gamma_sqmag)
def make_s21_fwhm(chi,sigma):
    val0 = s21_sqmag(0.0,chi,sigma)/2
    def s21_fwhm(detuning):
        result = s21_sqmag(detuning,chi,sigma) - val0
        return result
    return s21_fwhm



fwhm_array = np.zeros(n_sigmas)
for ii in range(n_sigmas):
    this_sigma = sigma_array[ii]
    detuning1 = sp.optimize.fsolve(make_s21_fwhm(chi,this_sigma),1/2,\
                                   xtol=tolerance,\
                                   maxfev=n_evals,\
                                   factor=0.001)
    #detuning1 = sol1.x
    detuning2 = sp.optimize.fsolve(make_s21_fwhm(chi,this_sigma),-1/2,\
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
plt.plot(sigma_array,fwhm_array,'.')
plt.plot(sigma_array,fwhm_fit)
plt.show()
