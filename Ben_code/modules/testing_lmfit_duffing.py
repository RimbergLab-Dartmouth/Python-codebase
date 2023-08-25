# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:34:17 2019

@author: Ben
"""

import numpy as np

def make_s11_duffing2(imag_cutoff=1E-20, \
                     sigma_range = 4.0, \
                     bistability_method='hysteresis',\
                     power_units = 'W'):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    def s11_duffing(f, \
                    f0, \
                    g_tot, \
                    chi, \
                    theta, \
                    sigma, \
                    duff, \
                    Pvna, \
                    atten):
        if power_units == 'dbm' or power_units == 'dBm':
            Pin = 10**(((Pvna + atten)-30.0)/10.0)
        elif power_units == 'w' or power_units == 'W':
            Pin = Pvna*atten
        f_step = f[1]-f[0] #ASSUMES UNIFORM SPACING
        ind_range = int(round((sigma*sigma_range)/f_step))
        start_arr = f[0]+(f_step*np.linspace(-1*ind_range,-1,ind_range))
        end_arr = f[-1]+(f_step*np.linspace(1,ind_range,ind_range))
        longer_f = np.hstack((start_arr,f,end_arr))
        #print(f)
        #print(longer_f)
        #input()
        response = np.zeros_like(longer_f)
        prev_root = None
        for ii in range(len(longer_f)):
            freq = longer_f[ii]
            coeffs = np.array([1.0, \
                               (2*(f0-freq))/duff, \
                               (((freq-f0)**2)+(((g_tot)**2)/4))/((duff**2)), \
                               (-1*((chi*g_tot)/(1+chi))*Pin)/(2*np.pi*h*freq*(duff**2))])
            roots = np.roots(coeffs)
            im_part = np.imag(roots)
            real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
            if len(real_roots) > 1:
                if bistability_method == 'hysteresis':
                    if prev_root:
                        root = real_roots[np.argmin(np.abs(real_roots-prev_root))]
                    else:
                        avg_root = 0
                        norm = 0
                        for this_root in real_roots:
                            energy = (h*f0*this_root)+(h*duff*(this_root**2))
                            weight = np.exp((-1*energy)/kT)
                            avg_root += this_root*weight
                            norm += weight
                        root = avg_root/norm
                elif bistability_method == 'boltzmann':
                    avg_root = 0
                    norm = 0
                    for this_root in real_roots:
                        energy = (h*f0*this_root)+(h*duff*(this_root**2))
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
        df = longer_f-f0-(duff*response)
        gamma = (df-((1j*g_tot*((1-chi)/(1+chi)))/2))/(df-((1j*g_tot)/2))
        x_axis = np.linspace(-f_step*ind_range,f_step*ind_range,(2*ind_range+1))
        gauss = (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-(x_axis)**2)/(2*(sigma**2)))
        avg_gamma = np.zeros(len(f),dtype=np.complex128)
        for ii in range(len(f)):
            avg_gamma[ii] = f_step*np.sum(gamma[ii:ii+(2*ind_range)+1]*gauss)
        #fig = plotr.plot_reflection_coefficient(f,avg_gamma)
        #plt.show()
        #print('duff = '+str(duff))
        #print('Pin = '+str(Pin))
        avg_gamma = ((avg_gamma-1)*np.exp(1j*theta)) + 1
        #result = np.hstack((np.real(avg_gamma),np.imag(avg_gamma)))
        return avg_gamma
    return s11_duffing


def make_s11_duffing(imag_cutoff=1E-20, \
                     sigma_range = 4.0, \
                     bistability_method='hysteresis',\
                     power_units = 'W'):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    def s11_duffing(f, \
                    f0, \
                    g_int, \
                    g_ext, \
                    theta, \
                    sigma, \
                    duff, \
                    Pin):
        if power_units == 'dbm':
            Pin = 10**((Pin-30.0)/10.0)
            #print(Pin)
        f_step = f[1]-f[0] #ASSUMES UNIFORM SPACING
        ind_range = int(round((sigma*sigma_range)/f_step))
        start_arr = f[0]+(f_step*np.linspace(-1*ind_range,-1,ind_range))
        end_arr = f[-1]+(f_step*np.linspace(1,ind_range,ind_range))
        longer_f = np.hstack((start_arr,f,end_arr))
        #print(f)
        #print(longer_f)
        #input()
        response = np.zeros_like(longer_f)
        prev_root = None
        for ii in range(len(longer_f)):
            freq = longer_f[ii]
            coeffs = np.array([1.0, \
                               (2*(f0-freq))/duff, \
                               (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/((duff**2)), \
                               (-g_ext*Pin)/(2*np.pi*h*freq*(duff**2))])
            roots = np.roots(coeffs)
            im_part = np.imag(roots)
            real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
            if len(real_roots) > 1:
                if bistability_method == 'hysteresis':
                    if prev_root:
                        root = real_roots[np.argmin(np.abs(real_roots-prev_root))]
                    else:
                        avg_root = 0
                        norm = 0
                        for this_root in real_roots:
                            energy = (h*f0*this_root)+(h*duff*(this_root**2))
                            weight = np.exp((-1*energy)/kT)
                            avg_root += this_root*weight
                            norm += weight
                        root = avg_root/norm
                elif bistability_method == 'boltzmann':
                    avg_root = 0
                    norm = 0
                    for this_root in real_roots:
                        energy = (h*f0*this_root)+(h*duff*(this_root**2))
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
        df = longer_f-f0-(duff*response)
        gamma = (df-((1j*(g_int-g_ext))/2))/(df-((1j*(g_int+g_ext))/2))
        x_axis = np.linspace(-f_step*ind_range,f_step*ind_range,(2*ind_range+1))
        gauss = (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-(x_axis)**2)/(2*(sigma**2)))
        avg_gamma = np.zeros(len(f),dtype=np.complex128)
        for ii in range(len(f)):
            avg_gamma[ii] = f_step*np.sum(gamma[ii:ii+(2*ind_range)+1]*gauss)
        #fig = plotr.plot_reflection_coefficient(f,avg_gamma)
        #plt.show()
        #print('duff = '+str(duff))
        #print('Pin = '+str(Pin))
        avg_gamma = ((avg_gamma-1)*np.exp(1j*theta)) + 1
        #result = np.hstack((np.real(avg_gamma),np.imag(avg_gamma)))
        return avg_gamma
    return s11_duffing
