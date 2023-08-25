# -*- coding: utf-8 -*-
"""
Created on Wed May  8 23:23:50 2019

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

chi=1.0
g_ext = chi/(1+chi)

n_freqs = 4000
freq_start = -25
freq_stop = 25
freqs  = np.linspace(freq_start,freq_stop,n_freqs)

n_sigmas = 100
sigma_start = 0.03
sigma_stop = 10.0
sigma_array = np.linspace(sigma_start,sigma_stop,n_sigmas)

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
            
            

"""          
plt.plot(sigma_array,chi_array,'.')
plt.title('chi vs sigma')
#plt.yscale('log')
#plt.xscale('log')
plt.show()

plt.plot(sigma_array,g_tot_array,'.')
plt.title('g tot vs sigma')
plt.show()

fit_ind = int(round(n_sigmas/5))
coeffs = np.polyfit(sigma_array[fit_ind:],g_tot_array[fit_ind:],1)
print('slope = '+str(coeffs[0]))
print('intercept = '+str(coeffs[1]))
g_tot_fit = (coeffs[0]*sigma_array)+coeffs[1]
plt.plot(sigma_array,g_tot_array,'.')
plt.plot(sigma_array,g_tot_fit)
plt.title('total damping rate + linear fit')
plt.show()
"""
"""
def exp_decay(time,rate,initial):
    result = initial*np.exp(-1*rate*time)
    return result

decay_model = lmfit.Model(exp_decay)
params = decay_model.make_params()
params['rate'].value = 1.0
params['initial'].value = chi
result = decay_model.fit(chi_array,params,time=sigma_array)
print(result.fit_report())
plt.plot(sigma_array,chi_array,'.')
plt.plot(sigma_array,result.best_fit)
plt.title('chi vs sigma')
#plt.yscale('log')
#plt.xscale('log')
plt.show()
"""
"""
def power_law(x,proportionality,exponent):
    result = proportionality*(x**exponent)
    return result

power_law_model = lmfit.Model(power_law)
params = power_law_model.make_params()
params['proportionality'].value = chi
params['exponent'].value = -1.0
fit_ind = int(round(n_sigmas/5))
result = power_law_model.fit(chi_array[fit_ind:],params,x=sigma_array[fit_ind:])
print(result.fit_report())
best_fit = power_law(sigma_array,result.params['proportionality'].value,result.params['exponent'].value)
plt.plot(sigma_array,chi_array,'.')
plt.plot(sigma_array,best_fit)
plt.title('chi vs sigma')
plt.yscale('log')
plt.xscale('log')
plt.show()

power_law_model = lmfit.Model(power_law)
params = power_law_model.make_params()
params['proportionality'].value = 2.0
params['exponent'].value = 1.0
fit_ind = int(round(n_sigmas/5))
result = power_law_model.fit(g_tot_array[fit_ind:],params,x=sigma_array[fit_ind:])
print(result.fit_report())
best_fit = power_law(sigma_array,result.params['proportionality'].value,result.params['exponent'].value)
plt.plot(sigma_array,g_tot_array,'.')
plt.plot(sigma_array,best_fit)
plt.title('g tot vs sigma (power law fit)')
plt.yscale('log')
plt.xscale('log')
plt.show()



plt.plot(sigma_array,chi_array,'.')
plt.title('chi vs sigma')
plt.yscale('log')
plt.xscale('log')
plt.show()

plt.plot(sigma_array,g_tot_array,'.')
plt.title('g tot vs sigma (power law fit)')
plt.yscale('log')
plt.xscale('log')
plt.show()


lowylim = min(np.amin(g_tot_array),np.amin(chi_array))
highylim = max(np.amax(g_tot_array),np.amax(chi_array))

colors = ['tab:blue','tab:red']

fig,ax1 = plt.subplots()
ax1.set_xlabel(r'$\sigma_{\omega_{0}}$')
ax1.set_ylabel(r'$\chi$',color=colors[0])
ax1.plot(sigma_array,chi_array,'.',color=colors[0])
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.tick_params(axis='y', labelcolor=colors[0])
ax1.set_ylim([lowylim,highylim])

ax2 = ax1.twinx()
ax2.set_ylabel(r'$\kappa_{\mathrm{tot}}$',color=colors[1])
ax2.plot(sigma_array,g_tot_array,'.',color=colors[1])
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.tick_params(axis='y', labelcolor=colors[1])
ax2.set_ylim([lowylim,highylim])

plt.show()
"""

lowylim = min(np.amin(g_tot_array),np.amin(chi_array))
highylim = max(np.amax(g_tot_array),np.amax(chi_array))

colors = ['tab:blue','tab:red']
labels = [r'$\kappa_{\mathrm{tot}}^{\prime}/\kappa_{\mathrm{tot}}$',r'$\xi^{\prime}$']

markers = ['o','s']

scale  =  10
markersize = 4
labelsize = 30
tickwidth = 2
ticklength = 8
ticklabelsize = 24
legendfontsize = labelsize

aspect_ratio = np.log10(sigma_array[-1]-sigma_array[0])/np.log10(highylim-lowylim)

fig,ax = plt.subplots(figsize=(scale,aspect_ratio*scale))
ax.set_aspect(aspect_ratio)
ax.set_xlabel(r'$\sigma_{\omega_{0}}/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
g_tot_plot = ax.plot(sigma_array,g_tot_array,markers[0],color=colors[1],label=labels[0],markersize=markersize)
chi_plot = ax.plot(sigma_array,chi_array,markers[1],color=colors[0],label=labels[1],markersize=markersize)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([lowylim,highylim])
ax.tick_params(width=tickwidth,length=ticklength,labelsize=ticklabelsize)
ax.tick_params(which='minor',width=tickwidth/2,length=ticklength/2)

ax.grid(linestyle='dotted',alpha=0.8)
ax.grid(which='minor',linestyle='--',alpha=0.8)

#legend = ax.legend(loc='upper left',fontsize=legendfontsize)
legend = ax.legend(loc='upper left',fontsize=legendfontsize,fancybox=True,framealpha=1.0)
#legend.legendHandles[0]._legmarker.set_markersize(6)
#legend.legendHandles[1]._legmarker.set_markersize(6)
legend.get_frame().set_facecolor('gainsboro')


plt.tight_layout()
plt.savefig(figs_dir+'s21_fitting_simulation.pdf',transparent=True)
plt.show()
