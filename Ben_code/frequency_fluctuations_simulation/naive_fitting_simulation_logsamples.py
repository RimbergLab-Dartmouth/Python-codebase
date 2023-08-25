# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:39:13 2019

@author: Ben
"""

import sys
import numpy as np
import scipy as sp
import lmfit
import matplotlib.pyplot as plt
import plotting_routines as plotr

figs_dir = 'figs/'

plot_flag = False
print_flag = False

chi=1.0
g_ext = chi/(1+chi)

n_freqs = 4000
freq_start = -25
freq_stop = 25
freqs  = np.linspace(freq_start,freq_stop,n_freqs)

n_sigmas = 30
sigma_start = 0.065
sigma_stop = 10.0
# params for finding threshold of 10% deviation between actual & apparent k_tot
#n_sigmas = 100
#sigma_start = 0.1
#sigma_stop = 0.2
sigma_array = np.logspace(np.log10(sigma_start),np.log10(sigma_stop),n_sigmas)

fit_ind = int(round(n_sigmas/5))

tolerance = 1.0E-13
n_evals = int(round(1.0E6))

def s21_model(df,sigma):
    term = (df-(1j/2))/(sigma*np.sqrt(2))
    #term = term.astype(np.complex128)
    term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))
    term2 = 1j*np.exp(-1*(term**2))
    term3 = (2/np.sqrt(np.pi))*sp.special.dawsn(term)
    s21 = (term1*(term2+term3))
    return s21

def s21_naive(df, chi, g_tot):
    s21 = (1j*(chi/(1+chi)))/((df/g_tot)-(1j/2))
    return s21

model = lmfit.Model(s21_naive)
params = model.make_params()
params['chi'].value = chi
params['chi'].min = 0.0
params['chi'].max = 100.0
params['g_tot'].value = 1.0
params['g_tot'].min = 0.0
params['g_tot'].max = 100.0

chi_array = np.zeros(n_sigmas)
g_tot_array = np.zeros(n_sigmas)
for ii in range(n_sigmas):
    this_sigma = sigma_array[ii]
    s21 = s21_model(freqs,this_sigma)
    result = model.fit(s21,params,df=freqs)
    chi_array[ii] = result.params['chi'].value
    g_tot_array[ii] = result.params['g_tot'].value
    if plot_flag:
        print(result.fit_report())
        fig = plotr.plot_reflection_coefficient_fits(freqs,s21,result.best_fit)
        plt.show()
        stop_flag = input('stop?')
        if stop_flag:
            sys.exit()
            
if print_flag:
    print('sigmas:')
    print(sigma_array)
    print('\n')
    print('kappa tots:')
    print(g_tot_array)
    print('\n')
    print('chis:')
    print(chi_array)
    print('\n')


lowxlim = sigma_array[0]*0.9
highxlim = sigma_array[-1]*1.1

lowylim = min(np.amin(g_tot_array),np.amin(chi_array))*0.9
highylim = max(np.amax(g_tot_array),np.amax(chi_array))*1.1

colors = ['tab:blue','tab:red']
labels = [r'$\kappa_{\mathrm{tot}}^{\prime}/\kappa_{\mathrm{tot}}$',r'$\xi^{\prime}$']

markers = ['o','^']

scale  =  8
markersize = 12
labelsize = 36
tickwidth = 2
ticklength = 10
ticklabelsize = 28
legendfontsize = labelsize

aspect_ratio = np.log10(highxlim-lowxlim)/np.log10(highylim-lowylim)

fig,ax = plt.subplots(figsize=(scale,1.7*aspect_ratio*scale))
#ax.set_aspect(aspect_ratio)
ax.set_aspect('equal')
ax.set_xlabel(r'$\sigma_{\omega_{0}}/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
g_tot_plot = ax.plot(sigma_array,g_tot_array,markers[0],color=colors[1],label=labels[0],markersize=markersize)
chi_plot = ax.plot(sigma_array,chi_array,markers[1],color=colors[0],label=labels[1],markersize=markersize)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([lowylim,highylim])
ax.set_xlim([lowxlim,highxlim])
ax.tick_params(width=tickwidth,length=ticklength,labelsize=ticklabelsize)
ax.tick_params(which='minor',width=tickwidth/2,length=ticklength/2)

ax.grid(linestyle='dashed',alpha=1.0)
ax.grid(which='minor',linestyle='dashed',alpha=1.0)

#legend = ax.legend(loc='upper left',fontsize=legendfontsize)
legend = ax.legend(loc='upper left',fontsize=legendfontsize,fancybox=True,framealpha=1.0)
#legend.legendHandles[0]._legmarker.set_markersize(6)
#legend.legendHandles[1]._legmarker.set_markersize(6)
legend.get_frame().set_facecolor('gainsboro')



plt.tight_layout()
plt.savefig(figs_dir+'s21_fitting_simulation_logsamples.pdf',transparent=True)
plt.show()
