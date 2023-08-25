# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:28:10 2019

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def duffing_discriminant(coeffs):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]
    discriminant = 18*a*b*c*d - \
                    4*(b**3)*d + \
                    (b**2)*(c**2) - \
                    4*a*(c**3) - \
                    27*(a**2)*(d**2)
    return discriminant

def gamma_duffing(df, chi, kerr, Nin, imag_cutoff=1E-30):
    response = []
    discriminants = []
    bistability_flag = 0
    prev_root = None
    for freq in df:
        coeffs = np.array([kerr**2, \
                           -2*kerr*freq, \
                           (freq**2)+(1/4), \
                           -1*(chi/(1+chi))*Nin])
        discriminants.append(duffing_discriminant(coeffs))
        roots = np.roots(coeffs)
#        print(roots)
        im_part = np.imag(roots)
        real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
#        print('im part: ')
#        print(im_part)
#        print('real roots:')
#        print(real_roots)
#        input()
        if len(real_roots) > 1:
            if bistability_flag == 0:
                print('MORE THAN ONE REAL ROOT!')
                bistability_flag = 1
            if prev_root:
                root = real_roots[np.argmin(np.abs(real_roots-prev_root))]
            else:
                root = real_roots[np.argmin(real_roots)]
        else:
            root = real_roots[0]
        response.append(root)
        prev_root = root
    response = np.array(response)
    discriminants = np.array(discriminants)
    numerator = (df-(kerr*response))-(1j*(((1-chi)/(1+chi))/2))
    denominator = (df-(kerr*response))-(1j/2)
    gamma = numerator/denominator
    return gamma, discriminants


scale = 12
ticklabelsize = 14
labelsize = 16
plotlabelsize = 20

freq_lim = 5
freq_stride = 100
df = np.linspace(-1*freq_lim,freq_lim,10001)
chi = 0.9
Nins = [10,20,30]
kerr = -0.1
if Nins[0]==0:
    labels = [r'\textbf{(a)} $P_{in} = '+str(Nins[0])+'$', \
                        r'\textbf{(b)} $P_{in} = '+str(Nins[1])+'\hbar\omega_{0}\kappa_{tot}$', \
                        r'\textbf{(c)} $P_{in} = '+str(Nins[2])+'\hbar\omega_{0}\kappa_{tot}$']
else:
    labels = [r'\textbf{(a)} $P_{in} = '+str(Nins[0])+'\hbar\omega_{0}\kappa_{tot}$', \
                        r'\textbf{(b)} $P_{in} = '+str(Nins[1])+'\hbar\omega_{0}\kappa_{tot}$', \
                        r'\textbf{(c)} $P_{in} = '+str(Nins[2])+'\hbar\omega_{0}\kappa_{tot}$']

discriminant_array = np.zeros((len(Nins),len(df)),dtype=float)
fig,axarr = plt.subplots(3,3,figsize=(scale,scale))
for ii in range(len(Nins)):
    label = labels[ii]
    Nin = Nins[ii]
    gamma,discriminants = gamma_duffing(df,chi,kerr,Nin)
    discriminant_array[ii] = discriminants
    x = np.real(gamma)
    x[np.where(x==0)]=+0.0
    y = np.imag(gamma)
    mag = np.sqrt((x**2)+(y**2))
    arg = np.arctan2(y,x)

    axarr[ii,0].text(-0.2, 1.2, label, transform=axarr[ii,0].transAxes,
      fontsize=plotlabelsize, fontweight='bold', va='top')

    
    axarr[ii,0].plot(x[0::freq_stride],y[0::freq_stride],linestyle='None',marker='.',color='r')
    axarr[ii,0].set_xlabel('Re[$\Gamma$]',fontsize=labelsize)
    axarr[ii,0].set_ylabel('Im[$\Gamma$]',fontsize=labelsize)
    axarr[ii,0].set_xlim((0,1.0))
    axarr[ii,0].set_ylim((-0.5,0.5))
    axarr[ii,0].set_aspect('equal')
    axarr[ii,0].set_yticks([-1/2,-1/4, 0, 1/4, 1/2])
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
plt.savefig('bistable_nonlinear_reflection.png')
plt.show()
