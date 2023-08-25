# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:33:24 2019

@author: Ben
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import lmfit

figs_dir = 'figs/'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

chi = 1.0

n_sigmas = 100
sigma_start = 0.1
sigma_stop = 20.0
sigma_array = np.linspace(sigma_start,sigma_stop,n_sigmas)

def s21_noisy(df,chi,sigma):
    argument = (df-(1j/2.0))/(sigma*np.sqrt(2.0))
    factor = (-1.0/sigma)*(np.sqrt(chi)/(1+chi))
    term1 = np.sqrt(np.pi/2.0)*np.exp(-1.0*(argument**2))
    term2 = -1j*np.sqrt(2.0)*sp.special.dawsn(argument)
    s21 = factor*(term1+term2)
    return s21

def s21_ideal(df, chi):
    s21 = ((-1.0*np.sqrt(chi))/(1+chi))*(1.0/((1j*df)+0.5))
    return s21

def simulated_chi(z):
    result = (2-(z**2)-(2*np.sqrt(1-(z**2))))/(z**2)
    return result

chi_array = np.zeros(n_sigmas)
for ii in range(n_sigmas):
    this_sigma = sigma_array[ii]
    s21_0 = np.abs(s21_noisy(0.0,chi,this_sigma))
    chi_array[ii] = simulated_chi(s21_0)
    
    
def power_law(x,proportionality,exponent):
    result = proportionality*(x**exponent)
    return result

power_law_model = lmfit.Model(power_law)
params = power_law_model.make_params()
params['proportionality'].value = chi
params['exponent'].value = -1.0
fit_ind = int(round(n_sigmas/3))
result = power_law_model.fit(chi_array[fit_ind:],params,x=sigma_array[fit_ind:])
print(result.fit_report())
best_fit = power_law(sigma_array,result.params['proportionality'].value,result.params['exponent'].value)
    
    
scale = 12
ticklabelsize = 20
labelsize = 24
plotlabelsize = 20
    
    
plt.figure(figsize=(scale,scale))
plt.plot(sigma_array,chi_array,'.')
plt.plot(sigma_array,best_fit)
plt.xlabel('$\sigma/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
plt.ylabel('$\chi$',fontsize=labelsize)
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = ticklabelsize)
ax.tick_params(axis = 'both', which = 'minor', labelsize = ticklabelsize)
plt.savefig(figs_dir+'chi_simulation.png')
plt.show()