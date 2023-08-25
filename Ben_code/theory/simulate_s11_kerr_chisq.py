# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:30:10 2020

@author: Ben
"""

import numpy as np
import matplotlib.pyplot as plt
import plotting_routines as plotr
import scipy as sp

def make_s11_kerr(imag_cutoff=1E-20, \
                     averaging_range = 5.0, \
                     bistability_method='boltzmann',\
                     power_units = 'dBm',
                     n_avg_mode = 'selfconsistent'):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    def s11_kerr(f,
                f0,
                ktot,
                xi,
                theta,
                kerr,
                Pvna,
                atten):
        if power_units == 'dbm' or power_units == 'dBm':
            Pin = 10**(((Pvna + atten)-30.0)/10.0)
        elif power_units == 'w' or power_units == 'W':
            Pin = Pvna*atten
        f_step = f[1]-f[0] #ASSUMES UNIFORM SPACING
        ind_range = int(round((abs(kerr)*averaging_range)/f_step))
        if kerr > 0:
            end_arr = f[-1]+(f_step*np.linspace(1,ind_range,ind_range))
            longer_f = np.hstack((f,end_arr))
        elif kerr < 0:
            start_arr = f[0]+(f_step*np.linspace(-1*ind_range,-1,ind_range))
            longer_f = np.hstack((start_arr,f))
        else:
            print('Kerr = 0!! \n Code not designed for this case.')

        #print(f)
        #print(longer_f)
        #input()
        response = np.zeros_like(longer_f)
        prev_root = None
        for ii in range(len(longer_f)):
            freq = longer_f[ii]
            coeffs = np.array([1.0, \
                               (2*(f0-freq))/kerr, \
                               (((freq-f0)**2)+(((ktot)**2)/4))/((kerr**2)), \
                               (-1*((xi*ktot)/(1+xi))*Pin)/(2*np.pi*h*freq*(kerr**2))])
            roots = np.roots(coeffs)
            im_part = np.imag(roots)
            real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
            if len(real_roots) > 1:
                #print('BISTABILITY THRESHOLD REACHED!')
                if bistability_method == 'hysteresis':
                    if prev_root:
                        root = real_roots[np.argmin(np.abs(real_roots-prev_root))]
                    else:
                        avg_root = 0
                        norm = 0
                        for this_root in real_roots:
                            if this_root == real_roots[1]:
                                continue
                            energy = (h*f0*this_root)+(h*kerr*(this_root**2))
                            weight = np.exp((-1*energy)/kT)
                            avg_root += this_root*weight
                            norm += weight
                        root = avg_root/norm
                elif bistability_method == 'boltzmann':
                    avg_root = 0
                    norm = 0
                    for this_root in real_roots:
                        # real_roots[0] and real_roots[2] stable
                        # real_roots[1] unstable
                        if this_root == real_roots[1]:
                            continue
                        energy = (h*f0*this_root)+(h*kerr*(this_root**2))
                        weight = np.exp((-1*energy)/kT)
                        avg_root += this_root*weight
                        norm += weight
                    root = avg_root/norm
                else:
                    print('INVALID METHOD OF DEALING WITH BISTABILITY!')
            else:
                root = real_roots[0]
            response[ii] = root
            prev_root = root
        df = longer_f-f0-(kerr*response)
        #plt.plot(longer_f,response)
        #plt.show()
        gamma = (df-((1j*ktot*((1-xi)/(1+xi)))/2))/(df-((1j*ktot)/2))
        avg_gamma = np.zeros(len(f),dtype=np.complex128)
        if kerr > 0:
            x_axis = np.linspace(0,f_step*ind_range,ind_range+1)
            n_avgs = response[0:len(f)]
        elif kerr < 0:
            x_axis = np.linspace(-1*f_step*ind_range,0,ind_range+1)
            n_avgs = response[-len(f):]
        for ii in range(len(f)):

            if n_avg_mode == 'single':
                n_avg = 2*(xi/(1+xi))*(Pin/(h*f0*2*np.pi*ktot))
            elif n_avg_mode == 'response':
                n_avg = n_avgs[ii]
            elif n_avg_mode == 'selfconsistent':
                n_avg = n_avgs[ii]
                noncentral_chisq = (2/np.abs(kerr))*np.exp(-2*(n_avg+(x_axis/kerr)))*sp.special.iv(0,np.sqrt((16*n_avg*x_axis)/kerr))
                norm = f_step*np.sum(noncentral_chisq)
                n_avg = (f_step*np.sum(response[ii:ii+ind_range+1]*noncentral_chisq))/norm
            noncentral_chisq = (2/np.abs(kerr))*np.exp(-2*(n_avg+(x_axis/kerr)))*sp.special.iv(0,np.sqrt((16*n_avg*x_axis)/kerr))
            norm = f_step*np.sum(noncentral_chisq)
            avg_gamma[ii] = (f_step*np.sum(gamma[ii:ii+ind_range+1]*noncentral_chisq))/norm

        #fig = plotr.plot_reflection_coefficient(f,avg_gamma)
        #plt.show()
        #print('duff = '+str(duff))
        #print('Pin = '+str(Pin))
        avg_gamma = ((avg_gamma-1)*np.exp(1j*theta)) + 1
        #result = np.hstack((np.real(avg_gamma),np.imag(avg_gamma)))
        return avg_gamma
    return s11_kerr


f0 = 5.728E9
kerr = 0.47E6
Pvna = -48.0
atten = -80.0
theta = 0

ktot0 = 1.29E6
xi0 = 5.4
kext0 = ktot0*(xi0/(1+xi0))
kint0 = ktot0/(1+xi0)

kext = kext0
kint = 3*kint0
ktot = kint+kext
xi = kext/kint

detuning_range = 8
n_detunings = 2000
f = f0 + (ktot*np.linspace(-1*detuning_range,detuning_range,n_detunings))

averaging_range = 30.0
f_step = f[1]-f[0]
ind_range = int(round((abs(kerr)*averaging_range)/f_step))
n_avg_mode = 'response'
bistability_method = 'boltzmann'
s11_kerr = make_s11_kerr(averaging_range=averaging_range,
                         bistability_method=bistability_method,
                         n_avg_mode=n_avg_mode)
s11 = s11_kerr(f,f0,ktot,xi,theta,kerr,Pvna,atten)

fig = plotr.plot_reflection_coefficient(f,s11)
plt.show()

"""
if kerr < 0:
    fig = plotr.plot_reflection_coefficient(f[0:ind_range+1],s11[0:ind_range+1])
    plt.show()
    
    fig = plotr.plot_reflection_coefficient(f[0:2*ind_range+1],s11[0:2*ind_range+1])
    plt.show()

else:
    fig = plotr.plot_reflection_coefficient(f[-ind_range:],s11[-ind_range:])
    plt.show()
    
    fig = plotr.plot_reflection_coefficient(f[-2*ind_range:],s11[-2*ind_range:])
    plt.show()

"""



