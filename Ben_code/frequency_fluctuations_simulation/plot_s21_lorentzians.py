# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:15:33 2019

@author: Ben
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import lmfit

figs_dir = 'figs/'

plot_sqmag = False
plot_dots = False
plot_dots_and_lines = False

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma = 1.0/(2.0)
chi = 1.0
g_ext = chi/(1+chi)

n_freqs = 1000
freq_lim = 5
freqs  = np.linspace(-1*freq_lim,freq_lim,n_freqs)
freq_stride = 2

def s21_noisy(df,chi,sigma):
    argument = (df-(1j/2.0))/(sigma*np.sqrt(2.0))
    factor = (-1.0/sigma)*(np.sqrt(chi)/(1+chi))
    term1 = np.sqrt(np.pi/2.0)*np.exp(-1.0*(argument**2))
    term2 = -1j*np.sqrt(2.0)*sp.special.dawsn(argument)
    s21 = factor*(term1+term2)
    return np.abs(s21)**2

def s21_ideal(df, chi):
    s21 = ((-1.0*np.sqrt(chi))/(1+chi))*(1.0/((1j*df)+0.5))
    return np.abs(s21)**2

def s21_naive(df, chi, g_tot):
    s21 = (1j*(chi/(1+chi)))/((df/g_tot)-(1j/2))
    return np.abs(s21)**2

df = freqs

s21 = s21_ideal(df,chi)
s21_noisy_data = s21_noisy(df,chi,sigma)

model = lmfit.Model(s21_naive)
params = model.make_params()
params['chi'].value = chi
params['chi'].min = 0.0
params['chi'].max = 100.0
params['g_tot'].value = 1.0
params['g_tot'].min = 0.0
params['g_tot'].max = 100.0
result = model.fit(s21_noisy_data,params,df=df)
print(result.fit_report())

s21_fit = result.best_fit

ideal_detunings = np.array([-0.5,0.5])
new_detunings = np.array([-1*result.params['g_tot'].value/2.0,result.params['g_tot'].value/2.0])

resonance_original = s21_ideal(0.0,1.0)
ideal_points_original = s21_ideal(ideal_detunings,1.0)
new_points_original = s21_ideal(new_detunings,1.0)

resonance_noisy = s21_noisy(0.0,1.0,sigma)
ideal_points_noisy = s21_noisy(ideal_detunings,1.0,sigma)
new_points_noisy = s21_noisy(new_detunings,1.0,sigma)

resonance_fit = s21_naive(0.0,result.params['chi'].value,result.params['g_tot'].value)
ideal_points_fit = s21_naive(ideal_detunings,result.params['chi'].value,result.params['g_tot'].value)
new_points_fit = s21_naive(new_detunings,result.params['chi'].value,result.params['g_tot'].value)

scale = 10
ticklabelsize = 20
labelsize = 22
plotlabelsize = 24
markersize = 10

colors = ['g','b','m']
markers = ['o','s','D']

fig = plt.figure(figsize=(scale,3*(scale-2)))
ax = fig.add_subplot(111,aspect='equal')

ax.plot(df,s21,color=colors[0])
ax.plot(df,s21_noisy_data,color=colors[1])
ax.plot(df,s21_fit,color=colors[2])

ax.plot(0.0,resonance_original,markers[0],color=colors[0],markersize=markersize)
ax.plot(ideal_detunings,ideal_points_original,markers[1],color=colors[0],markersize=markersize)
ax.plot(new_detunings,new_points_original,markers[2],color=colors[0],markersize=markersize)

ax.plot(0.0,resonance_noisy,markers[0],color=colors[1],markersize=markersize)
ax.plot(ideal_detunings,ideal_points_noisy,markers[1],color=colors[1],markersize=markersize)
ax.plot(new_detunings,new_points_noisy,markers[2],color=colors[1],markersize=markersize)

ax.plot(0.0,resonance_fit,markers[0],color=colors[2],markersize=markersize)
ax.plot(ideal_detunings,ideal_points_fit,markers[1],color=colors[2],markersize=markersize)
ax.plot(new_detunings,new_points_fit,markers[2],color=colors[2],markersize=markersize)


ax.set_xlabel('$\Delta/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
ax.set_ylabel('$|S_{21}|^{2}$',fontsize=labelsize)
ax.set_xlim((-1*freq_lim,freq_lim))
ax.set_ylim((0.0,1.0))
ax.set_yticks([0,1/4,1/2,3/4,1])
ax.tick_params(labelsize=ticklabelsize)
ax.grid(linestyle='dotted')
ax.set_aspect(2.0*freq_lim)


plt.savefig(figs_dir+'s21_lorentzians.png')
plt.show()