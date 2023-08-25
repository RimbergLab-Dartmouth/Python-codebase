# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:14:26 2019

@author: Ben
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def gamma_ideal(df,chi):
    # df = df/g_int
    # chi = g_ext/g_int
    numerator = df-(1j*(((1-chi)/(1+chi))/2))
    denominator = df-(1j/2)
    return numerator/denominator

scale = 12
ticklabelsize = 14
labelsize = 16
plotlabelsize = 20



freq_lim = 2
df = np.linspace(-1*freq_lim,freq_lim,1001)
chis = [1/3,1,3]
labels = [r'\textbf{(a)} $\chi = 1/3$', \
                    r'\textbf{(b)} $\chi = 1$', \
                    r'\textbf{(c)} $\chi = 3$']

fig,axarr = plt.subplots(3,3,figsize=(scale,scale))
for ii in range(len(chis)):
    label = labels[ii]
    chi = chis[ii]
    gamma = gamma_ideal(df,chi)
    x = np.real(gamma)
    x[np.where(x==0)]=+0.0
    y = np.imag(gamma)
    mag = np.sqrt((x**2)+(y**2))
    arg = np.arctan2(y,x)

    axarr[ii,0].text(-0.2, 1.2, label, transform=axarr[ii,0].transAxes,
      fontsize=plotlabelsize, fontweight='bold', va='top')

    
    axarr[ii,0].plot(x[0::20],y[0::20],linestyle='None',marker='.',color='r')
    axarr[ii,0].set_xlabel('Re[$\Gamma$]',fontsize=labelsize)
    axarr[ii,0].set_ylabel('Im[$\Gamma$]',fontsize=labelsize)
    axarr[ii,0].set_xlim((-0.55,1.05))
    axarr[ii,0].set_ylim((-0.8,0.8))
    axarr[ii,0].set_aspect('equal')
    axarr[ii,0].set_yticks([-3/4,-1/2,-1/4, 0, 1/4, 1/2, 3/4])
    axarr[ii,0].tick_params(labelsize=ticklabelsize)
    axarr[ii,0].grid(linestyle='dotted')

    axarr[ii,1].plot(df,mag,color='r')
    axarr[ii,1].set_xlabel('$\delta\omega/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
    axarr[ii,1].set_ylabel('$| \Gamma |$',fontsize=labelsize)
    axarr[ii,1].set_xlim((-1*freq_lim,freq_lim))
    axarr[ii,1].set_ylim((0.0,1.0))
    axarr[ii,1].set_yticks([0,1/4,1/2,3/4,1])
    axarr[ii,1].tick_params(labelsize=ticklabelsize)
    axarr[ii,1].grid(linestyle='dotted')

    axarr[ii,2].plot(df,arg,color='r')
    axarr[ii,2].set_xlabel('$\delta\omega/\kappa_{\mathrm{tot}}$',fontsize=labelsize)
    axarr[ii,2].set_ylabel('arg[$\Gamma$]',fontsize=labelsize)
    axarr[ii,2].set_xlim((-1*freq_lim,freq_lim))
    axarr[ii,2].set_ylim((-np.pi,np.pi))
    axarr[ii,2].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    axarr[ii,2].set_yticklabels(['$-\pi$','$-\pi/2$','0','$\pi/2$','$\pi$'])
    axarr[ii,2].tick_params(labelsize=ticklabelsize)
    axarr[ii,2].grid(linestyle='dotted')
    
plt.tight_layout()
plt.savefig('linear_reflection_visualization.png')
plt.show()
