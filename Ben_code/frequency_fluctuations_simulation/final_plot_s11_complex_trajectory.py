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

file_extension = 'eps'
dpi = 1000

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sigma = 1.0/(2.0)
xi = 1.0
g_ext = xi/(1+xi)

n_freqs = 100000
freq_lim = 200
freqs  = np.linspace(-1*freq_lim,freq_lim,n_freqs)
freq_stride = 2

def s11_noisy(df,xi,sigma):
    factor = (-1*np.sqrt(xi)*np.pi)/((1+xi)*np.sqrt(2*np.pi*(sigma**2)))
    argument = ((-2*df)+(1j))/(2*np.sqrt(2)*sigma)
    s11 = 1+(np.sqrt(xi)*factor*sp.special.wofz(argument))
    return s11

def s11_ideal(df, xi):
    s11 = 1-((2*xi)/((1+xi)*(1+((2j*df)))))
    return s11

def s11_naive(df, xi, g_tot):
    s21 = 1-((2*xi)/((1+xi)*(1+((2j*df)/g_tot))))
    return s21

df = freqs

s11 = s11_ideal(df,xi)
x = np.real(s11)
x[np.where(x==0)]=+0.0
y = np.imag(s11)
mag = np.sqrt((x**2)+(y**2))
mag_sq = (x**2)+(y**2)
arg = np.arctan2(y,x)%(2*np.pi)

s11_noisy_data = s11_noisy(df,xi,sigma)
x_noisy = np.real(s11_noisy_data)
x_noisy[np.where(x_noisy==0)]=+0.0
y_noisy = np.imag(s11_noisy_data)
mag_noisy = np.sqrt((x_noisy**2)+(y_noisy**2))
mag_sq_noisy = (x_noisy**2)+(y_noisy**2)
arg_noisy = np.arctan2(y_noisy,x_noisy)%(2*np.pi)

model = lmfit.Model(s11_naive)
params = model.make_params()
params['xi'].value = xi
params['xi'].min = 0.0
params['xi'].max = 100.0
params['g_tot'].value = 1.0
params['g_tot'].min = 0.0
params['g_tot'].max = 100.0
result = model.fit(s11_noisy_data,params,df=df)
print(result.fit_report())

s11_fit = result.best_fit
x_fit = np.real(s11_fit)
x_fit[np.where(x_fit==0)]=+0.0
y_fit = np.imag(s11_fit)
mag_fit = np.sqrt((x_fit**2)+(y_fit**2))
mag_sq_fit = (x_fit**2)+(y_fit**2)
arg_fit = np.arctan2(y_fit,x_fit)%(2*np.pi)

ideal_detunings = np.array([-0.5,0.5])
new_detunings = np.array([-1*result.params['g_tot'].value/2.0,result.params['g_tot'].value/2.0])

resonance_original = s11_ideal(0.0,xi)
ideal_points_original = s11_ideal(ideal_detunings,xi)
new_points_original = s11_ideal(new_detunings,xi)

resonance_noisy = s11_noisy(0.0,xi,sigma)
ideal_points_noisy = s11_noisy(ideal_detunings,xi,sigma)
new_points_noisy = s11_noisy(new_detunings,xi,sigma)

resonance_fit = s11_naive(0.0,result.params['xi'].value,result.params['g_tot'].value)
ideal_points_fit = s11_naive(ideal_detunings,result.params['xi'].value,result.params['g_tot'].value)
new_points_fit = s11_naive(new_detunings,result.params['xi'].value,result.params['g_tot'].value)


width = 3.375
aspect = 0.975
height = width*aspect
ticklabelsize = 8
labelsize = 10
markersize = 4
linewidth = 1.5

xlabelpad = 5
ylabelpad = 0
fig_pad = -0.15
fig_wpad = 0.0
fig_hpad = 0.0

bottom_adjust = 0.1
left_adjust = 0.13

ticklength = 3
tickwidth = 0.6

grid_linewidth = 0.5
grid_linestyle = 'solid'
grid_alpha = 1.0
grid_color = 'silver'

colors = ['tab:green','tab:blue','tab:red']
#colors = ['g','b','r']

linestyles = [':','-','--']
markers = ['o','s','^']

#fig = plt.figure(figsize=(width,height))
#ax = fig.add_subplot(111,aspect='equal')

fig,ax = plt.subplots(1,figsize=(width,height))
ax.set_aspect('equal')

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

ax.set_xlabel(r'Re[$S_{11}$]',fontsize=labelsize)
ax.set_ylabel(r'Im[$S_{11}$]',fontsize=labelsize)

ax.xaxis.labelpad = xlabelpad
ax.yaxis.labelpad = ylabelpad

ax.set_xlim((-0.025,1.025))
ax.set_ylim((-0.525,0.525))
ax.set_aspect('equal')
ax.set_yticks([-1/2,-1/4, 0, 1/4, 1/2])
ax.set_yticklabels([r'-$1/2$',r'-$1/4$',r'$0$',r'$1/4$',r'$1/2$'])
ax.set_xticks([0,1/4,1/2,3/4,1])
ax.set_xticklabels([r'$0$',r'$1/4$',r'$1/2$',r'$3/4$',r'$1$'])

ax.tick_params(labelsize=ticklabelsize,length=ticklength,width=tickwidth)
ax.grid(linestyle=grid_linestyle,alpha=grid_alpha,linewidth=grid_linewidth,color=grid_color)


#plt.tight_layout(pad=figpad)
#fig.subplots_adjust(bottom=bottom_adjust,left=left_adjust)
plt.tight_layout(pad=fig_pad,w_pad = fig_wpad,h_pad=fig_hpad)
fig.subplots_adjust(left=left_adjust)


plt.savefig(figs_dir+'final_s11_complex_trajectory.'+file_extension,transparent=True,dpi=dpi)
plt.show()