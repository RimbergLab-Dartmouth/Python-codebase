# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:22:03 2019

@author: Ben
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import lmfit

figs_dir = 'figs/'


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma = 1.0/(2.0)
chi = 2.0
g_ext = chi/(1+chi)

n_freqs = 100000
freq_lim = 200
freqs  = np.linspace(-1*freq_lim,freq_lim,n_freqs)
freq_stride = 2

def s21_noisy(df,chi,sigma):
    argument = (df-(1j/2.0))/(sigma*np.sqrt(2.0))
    factor = (-1.0/sigma)*(np.sqrt(chi)/(1+chi))
    term1 = np.sqrt(np.pi/2.0)*np.exp(-1.0*(argument**2))
    term2 = -1j*np.sqrt(2.0)*sp.special.dawsn(argument)
    s21 = factor*(term1+term2)
    return s21

def s21_ideal(df, chi):
    s21 = (-2*np.sqrt(chi))/((1+chi)*(1+((2j*df))))
    return s21

def s21_naive(df, chi, g_tot):
    s21 = (-2*np.sqrt(chi))/((1+chi)*(1+((2j*df)/g_tot)))
    return s21

df = freqs

s21 = s21_ideal(df,chi)
x = np.real(s21)
x[np.where(x==0)]=+0.0
y = np.imag(s21)
mag = np.sqrt((x**2)+(y**2))
mag_sq = (x**2)+(y**2)
arg = np.arctan2(y,x)%(2*np.pi)

s21_noisy_data = s21_noisy(df,chi,sigma)
x_noisy = np.real(s21_noisy_data)
x_noisy[np.where(x_noisy==0)]=+0.0
y_noisy = np.imag(s21_noisy_data)
mag_noisy = np.sqrt((x_noisy**2)+(y_noisy**2))
mag_sq_noisy = (x_noisy**2)+(y_noisy**2)
arg_noisy = np.arctan2(y_noisy,x_noisy)%(2*np.pi)

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
x_fit = np.real(s21_fit)
x_fit[np.where(x_fit==0)]=+0.0
y_fit = np.imag(s21_fit)
mag_fit = np.sqrt((x_fit**2)+(y_fit**2))
mag_sq_fit = (x_fit**2)+(y_fit**2)
arg_fit = np.arctan2(y_fit,x_fit)%(2*np.pi)

ideal_detunings = np.array([-0.5,0.5])
new_detunings = np.array([-1*result.params['g_tot'].value/2.0,result.params['g_tot'].value/2.0])

resonance_original = s21_ideal(0.0,chi)
ideal_points_original = s21_ideal(ideal_detunings,chi)
new_points_original = s21_ideal(new_detunings,chi)

resonance_noisy = s21_noisy(0.0,chi,sigma)
ideal_points_noisy = s21_noisy(ideal_detunings,chi,sigma)
new_points_noisy = s21_noisy(new_detunings,chi,sigma)

resonance_fit = s21_naive(0.0,result.params['chi'].value,result.params['g_tot'].value)
ideal_points_fit = s21_naive(ideal_detunings,result.params['chi'].value,result.params['g_tot'].value)
new_points_fit = s21_naive(new_detunings,result.params['chi'].value,result.params['g_tot'].value)

scale = 10
ticklabelsize = 24
labelsize = 30
markersize = 16
linewidth = 4

colors = ['tab:green','tab:blue','tab:red']
#colors = ['g','b','r']

linestyles = [':','-','--']
markers = ['o','s','^']

fig = plt.figure(figsize=(1.05*scale,scale))
ax = fig.add_subplot(111,aspect='equal')

ax.plot(x[0::freq_stride],y[0::freq_stride],color=colors[0],linestyle=linestyles[0],linewidth=linewidth)
ax.plot(x_noisy[0::freq_stride],y_noisy[0::freq_stride],color=colors[1],linestyle=linestyles[1],linewidth=linewidth)
ax.plot(x_fit[0::freq_stride],y_fit[0::freq_stride],color=colors[2],linestyle=linestyles[2],linewidth=linewidth)

ax.plot(np.real(resonance_original),np.imag(resonance_original),markers[0],color=colors[0],markersize=markersize)
ax.plot(np.real(ideal_points_original),np.imag(ideal_points_original),markers[1],color=colors[0],markersize=markersize)
ax.plot(np.real(new_points_original),np.imag(new_points_original),markers[2],color=colors[0],markersize=markersize)

ax.plot(np.real(resonance_noisy),np.imag(resonance_noisy),markers[0],color=colors[1],markersize=markersize)
ax.plot(np.real(ideal_points_noisy),np.imag(ideal_points_noisy),markers[1],color=colors[1],markersize=markersize)
ax.plot(np.real(new_points_noisy),np.imag(new_points_noisy),markers[2],color=colors[1],markersize=markersize)

ax.plot(np.real(resonance_fit),np.imag(resonance_fit),markers[0],color=colors[2],markersize=markersize)
ax.plot(np.real(ideal_points_fit),np.imag(ideal_points_fit),markers[1],color=colors[2],markersize=markersize)
ax.plot(np.real(new_points_fit),np.imag(new_points_fit),markers[2],color=colors[2],markersize=markersize)

ax.set_xlabel(r'Re[$S_{21}$]',fontsize=labelsize)
ax.set_ylabel(r'Im[$S_{21}$]',fontsize=labelsize)

ax.set_xlim((-1.025,0.025))
ax.set_ylim((-0.525,0.525))
ax.set_aspect('equal')
ax.set_yticks([-1/2,-1/4, 0, 1/4, 1/2])
ax.set_yticklabels(['-1/2','-1/4','0','1/4','1/2'])
ax.set_xticks([-1,-3/4, -1/2, -1/4, 0])
ax.set_xticklabels(['-1','-3/4','-1/2','-1/4','0'])

ax.tick_params(labelsize=ticklabelsize)
ax.grid(linestyle='dashed',alpha=1.0)

plt.tight_layout()
plt.savefig(figs_dir+'s21_complex_trajectory.pdf',transparent=True)
plt.show()