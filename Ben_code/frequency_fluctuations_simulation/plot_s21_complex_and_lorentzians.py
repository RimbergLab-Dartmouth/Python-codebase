# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:32:56 2019

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
    return s21

def s21_ideal(df, chi):
    s21 = ((-1.0*np.sqrt(chi))/(1+chi))*(1.0/((1j*df)+0.5))
    return s21

def s21_naive(df, chi, g_tot):
    s21 = (1j*(chi/(1+chi)))/((df/g_tot)-(1j/2))
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

resonance_original = s21_ideal(0.0,1.0)
ideal_points_original = s21_ideal(ideal_detunings,1.0)
new_points_original = s21_ideal(new_detunings,1.0)

resonance_noisy = s21_noisy(0.0,1.0,sigma)
ideal_points_noisy = s21_noisy(ideal_detunings,1.0,sigma)
new_points_noisy = s21_noisy(new_detunings,1.0,sigma)

resonance_fit = s21_naive(0.0,result.params['chi'].value,result.params['g_tot'].value)
ideal_points_fit = s21_naive(ideal_detunings,result.params['chi'].value,result.params['g_tot'].value)
new_points_fit = s21_naive(new_detunings,result.params['chi'].value,result.params['g_tot'].value)

scale = 15
ticklabelsize = 20
labelsize = 22
plotlabelsize = 24
markersize = 10

colors = ['g','b','r']

fig,axarr = plt.subplots(2,1,figsize=(scale,2*(scale-2)))
if plot_dots:
    axarr[0].plot(x[0::freq_stride],y[0::freq_stride],'.',color=colors[0])
    axarr[0].plot(x_noisy[0::freq_stride],y_noisy[0::freq_stride],'.',color=colors[1])
    axarr[0].plot(x_fit[0::freq_stride],y_fit[0::freq_stride],'.',color=colors[2])
elif plot_dots_and_lines:
    axarr[0].plot(x[0::freq_stride],y[0::freq_stride],linestyle='-',marker='o',color=colors[0])
    axarr[0].plot(x_noisy[0::freq_stride],y_noisy[0::freq_stride],linestyle='-',marker='o',color=colors[1])
    axarr[0].plot(x_fit[0::freq_stride],y_fit[0::freq_stride],linestyle='-',marker='o',color=colors[2])
else:
    axarr[0].plot(x[0::freq_stride],y[0::freq_stride],color=colors[0])
    axarr[0].plot(x_noisy[0::freq_stride],y_noisy[0::freq_stride],color=colors[1])
    axarr[0].plot(x_fit[0::freq_stride],y_fit[0::freq_stride],color=colors[2])

axarr[0].plot(np.real(resonance_original),np.imag(resonance_original),'o',color=colors[0],markersize=markersize)
axarr[0].plot(np.real(ideal_points_original),np.imag(ideal_points_original),'s',color=colors[0],markersize=markersize)
axarr[0].plot(np.real(new_points_original),np.imag(new_points_original),'^',color=colors[0],markersize=markersize)

axarr[0].plot(np.real(resonance_noisy),np.imag(resonance_noisy),'o',color=colors[1],markersize=markersize)
axarr[0].plot(np.real(ideal_points_noisy),np.imag(ideal_points_noisy),'s',color=colors[1],markersize=markersize)
axarr[0].plot(np.real(new_points_noisy),np.imag(new_points_noisy),'^',color=colors[1],markersize=markersize)

axarr[0].plot(np.real(resonance_fit),np.imag(resonance_fit),'o',color=colors[2],markersize=markersize)
axarr[0].plot(np.real(ideal_points_fit),np.imag(ideal_points_fit),'s',color=colors[2],markersize=markersize)
axarr[0].plot(np.real(new_points_fit),np.imag(new_points_fit),'^',color=colors[2],markersize=markersize)

axarr[0].set_xlabel('Re[$S_{21}$]',fontsize=labelsize)
axarr[0].set_ylabel('Im[$S_{21}$]',fontsize=labelsize)
axarr[0].set_xlim((-1.0,0.0))
axarr[0].set_ylim((-0.5,0.5))
axarr[0].set_aspect('equal')
axarr[0].set_yticks([-1/2,-1/4, 0, 1/4, 1/2])
#ax.set_xticks([-1,-3/4, -1/2, -1/4, 0])
axarr[0].tick_params(labelsize=ticklabelsize)
axarr[0].grid(linestyle='dotted')


axarr[1].plot(df,np.abs(s21)**2,color=colors[0])
axarr[1].plot(df,np.abs(s21_noisy_data)**2,color=colors[1])
axarr[1].plot(df,np.abs(s21_fit)**2,color=colors[2])

axarr[1].plot(0.0,np.abs(resonance_original)**2,'o',color=colors[0],markersize=markersize)
axarr[1].plot(ideal_detunings,np.abs(ideal_points_original)**2,'s',color=colors[0],markersize=markersize)
axarr[1].plot(new_detunings,np.abs(new_points_original)**2,'^',color=colors[0],markersize=markersize)

axarr[1].plot(0.0,np.abs(resonance_noisy)**2,'o',color=colors[1],markersize=markersize)
axarr[1].plot(ideal_detunings,np.abs(ideal_points_noisy)**2,'s',color=colors[1],markersize=markersize)
axarr[1].plot(new_detunings,np.abs(new_points_noisy)**2,'^',color=colors[1],markersize=markersize)

axarr[1].plot(0.0,np.abs(resonance_fit)**2,'o',color=colors[2],markersize=markersize)
axarr[1].plot(ideal_detunings,np.abs(ideal_points_fit)**2,'s',color=colors[2],markersize=markersize)
axarr[1].plot(new_detunings,np.abs(new_points_fit)**2,'^',color=colors[2],markersize=markersize)


axarr[1].set_xlabel('$\Delta/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
axarr[1].set_ylabel('$|S_{21}|^{2}$',fontsize=labelsize)
axarr[1].set_xlim((-1*freq_lim,freq_lim))
axarr[1].set_ylim((0.0,1.0))
axarr[1].set_yticks([0,1/4,1/2,3/4,1])
axarr[1].tick_params(labelsize=ticklabelsize)
axarr[1].grid(linestyle='dotted')
axarr[1].set_aspect(2.0*freq_lim)


plt.savefig(figs_dir+'s21_complex_and_lorentzians.png')
plt.show()