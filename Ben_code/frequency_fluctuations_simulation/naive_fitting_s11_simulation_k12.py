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

file_extension = 'eps'
dpi = 1000

plot_flag = False
print_flag = False

xi=10.0
xi_guess = 1.0

n_freqs = 10000
freq_start = -50
freq_stop = 50
freqs  = np.linspace(freq_start,freq_stop,n_freqs)

n_sigmas = 40
sigma_start = 0.01
sigma_stop = 10.0
# params for finding threshold of 10% deviation between actual & apparent k_tot
#n_sigmas = 100
#sigma_start = 0.1
#sigma_stop = 0.2
sigma_array = np.logspace(np.log10(sigma_start),np.log10(sigma_stop),n_sigmas)

fit_ind = int(round(n_sigmas/5))

tolerance = 1.0E-13
n_evals = int(round(1.0E6))

def s11_model(df,sigma):
    factor = (-1*np.sqrt(xi)*np.pi)/((1+xi)*np.sqrt(2*np.pi*(sigma**2)))
    argument = ((-2*df)+(1j))/(2*np.sqrt(2)*sigma)
    s11 = 1+(np.sqrt(xi)*factor*sp.special.wofz(argument))
    return s11

def s11_naive(df, xi, g_tot):
    s11 = 1-((2*xi)/((1+xi)*(1+((2j*df)/g_tot))))
    return s11

model = lmfit.Model(s11_naive)
params = model.make_params()
params['xi'].value = xi_guess
params['xi'].min = 0.0
params['xi'].max = 10000.0
params['g_tot'].value = 1.0
params['g_tot'].min = 0.0
params['g_tot'].max = 1000.0

xi_array = np.zeros(n_sigmas)
g_tot_array = np.zeros(n_sigmas)
for ii in range(n_sigmas):
    params['xi'].value = xi_guess
    this_sigma = sigma_array[ii]
    s11 = s11_model(freqs,this_sigma)
    result = model.fit(s11,params,df=freqs)
    xi_array[ii] = result.params['xi'].value
    g_tot_array[ii] = result.params['g_tot'].value
    if plot_flag:
        print(result.fit_report())
        fig = plotr.plot_reflection_coefficient_fits(freqs,s11,result.best_fit)
        plt.show()
        stop_flag = input('stop?')
        if stop_flag:
            sys.exit()
            
#xi_array = xi_array/xi
k1_array = (xi_array*g_tot_array)/(1+xi_array)
k2_array = g_tot_array/(1+xi_array)

if print_flag:
    print('sigmas:')
    print(sigma_array)
    print('\n')
    print('kappa tots:')
    print(g_tot_array)
    print('\n')
    print('xis:')
    print(xi_array)
    print('\n')

#xi_array = (xi_array/(1+xi_array))*((1+xi)/xi)

lowxlim = sigma_array[0]*0.92
highxlim = sigma_array[-1]*1.1

lowylim = min(np.amin(g_tot_array),np.amin(xi_array))*0.9
highylim = max(np.amax(g_tot_array),np.amax(xi_array))*1.1

colors = ['tab:blue','tab:red']
#colors = ['b','r']

labels = [r'$\kappa_{\mathrm{tot}}^{\prime}/\kappa_{\mathrm{tot}}$',r'$\xi^{\prime}$']

markers = ['o','^']

width = 3.375
aspect = 0.925
height = width*aspect
markersize = 3
labelsize = 11.0
tickwidth = 0.5
ticklength = 5
ticklabelsize = 10
legendfontsize = labelsize

grid_linewidth = 0.3
grid_linestyle = 'solid'
grid_alpha = 1.0
grid_color = 'silver'

xlabelpad = 1

fig_pad = -0.35
fig_wpad = 0.0
fig_hpad = 0.0

bottom_adjust = 0.25
left_adjust = 0.0925

#aspect_ratio = np.log10(highxlim-lowxlim)/np.log10(highylim-lowylim)

fig,ax = plt.subplots(figsize=(width,height))
ax.set_aspect('equal')
ax.set_xlabel(r'$\sigma_{\omega_{0}}/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
g_tot_plot = ax.plot(sigma_array,g_tot_array,markers[0],color=colors[1],label=labels[0],markersize=markersize)
xi_plot = ax.plot(sigma_array,xi_array,markers[1],color=colors[0],label=labels[1],markersize=markersize)
k1_plot = ax.plot(sigma_array,k1_array,label=r'$\kappa_{1}^{\prime}/\kappa_{\mathrm{tot}}$')
k2_plot = ax.plot(sigma_array,k2_array,label=r'$\kappa_{2}^{\prime}/\kappa_{\mathrm{tot}}$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([lowylim,highylim])
ax.set_xlim([lowxlim,highxlim])
ax.tick_params(width=tickwidth,length=ticklength,labelsize=ticklabelsize)
ax.tick_params(which='minor',width=tickwidth,length=ticklength/2)

ax.set_xticks([0.01,0.1,1.0,10.0])
ax.set_xticklabels([r'$10^{-2}$',r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$'])
ax.set_yticks([0.1,1.0,10.0])
ax.set_yticklabels([r'$10^{-1}$',r'$10^{0}$',r'$10^{1}$'])

#ax.set_xticks(ax.get_xticks()[1:])


ax.grid(linestyle=grid_linestyle,alpha=grid_alpha,linewidth=grid_linewidth,color=grid_color)
ax.grid(which='minor',linestyle=grid_linestyle,alpha=grid_alpha,linewidth=grid_linewidth,color=grid_color)

#legend = ax.legend(loc='upper left',fontsize=legendfontsize)
legend = ax.legend(loc='upper left',fontsize=legendfontsize,fancybox=True,framealpha=1.0)
#legend.legendHandles[0]._legmarker.set_markersize(6)
#legend.legendHandles[1]._legmarker.set_markersize(6)
legend.get_frame().set_facecolor('gainsboro')

ax.xaxis.labelpad = xlabelpad

plt.tight_layout(pad=fig_pad,w_pad=fig_wpad,h_pad=fig_hpad)
fig.subplots_adjust(left=left_adjust)

#plt.savefig(figs_dir+'final_s11_fitting_simulation_logsamples_k12.'+file_extension,transparent=True,dpi=dpi)
plt.show()
