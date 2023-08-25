# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:06:28 2019

@author: Ben
"""

import scipy as sp
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import plotting_routines as plotr

def convert_to_complex_gamma(mag, phase, mag_str = 'log', phase_str = 'deg'):
    # mag_str = 'log' or 'lin'
    # phase_str = 'deg' or 'rad'
    if mag_str == 'log':
        linmag = np.power(10.0,mag/20)
    elif mag_str == 'lin':
        linmag = mag
    if phase_str == 'deg':
        phase_rad = phase*(np.pi/180)
    elif phase_str == 'rad':
        phase_rad = phase
    x = linmag*np.cos(phase_rad)
    y = linmag*np.sin(phase_rad)
    gamma = x+(1j*y)
    return gamma

def invert_from_complex_gamma(gamma, mag_str = 'log', phase_str = 'deg'):
    # mag_str = 'log' or 'lin'
    # phase_str = 'deg' or 'rad'
    x = np.real(gamma)
    y = np.imag(gamma)
    mag = np.sqrt((x**2)+(y**2))
    phase = np.arctan2(y,x)
    if mag_str == 'log':
        mag = 20*np.log10(mag)
    if phase_str == 'deg':
        phase = (180.0/np.pi)*phase
    return mag,phase
    
def rotate_gamma(gamma,theta):
    rotated_gamma = ((gamma-1.0)*np.exp(-1j*theta))+1.0
    return rotated_gamma


def make_s11(model, fixed_params = None):
    # model = string indicating model to use
    if model == 'ideal':
        def s11_model(f,f0,g_int,g_ext):
            # complex reflection coefficient
            df = f-f0
            numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
            denominator = (4*(df**2))+((g_int + g_ext)**2)
            gamma = numerator/denominator
            result = np.hstack((np.real(gamma),np.imag(gamma)))
            return result
        
    elif model == 'rotated':
        def s11_model(f,f0,g_int,g_ext,theta):
            # add an empirical parameter for the phase of the internal field
            df = f-f0
            numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
            denominator = (4*(df**2))+((g_int + g_ext)**2)
            gamma = numerator/denominator
            gamma = ((gamma-1)*np.exp(1j*theta)) + 1
            result = np.hstack((np.real(gamma),np.imag(gamma)))
            return result
        
    elif model == 'noisy':
        def s11_model(f,f0,g_int,g_ext,theta,sigma):
            f = f.astype(np.float64)
            df = f-f0
            term = (df-(1j*((g_int+g_ext)/2)))/(sigma*np.sqrt(2))
            term = term.astype(np.complex128)
            term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))
            term2 = 1j*np.exp(-1*(term**2))
            term3 = (2/np.sqrt(np.pi))*sp.special.dawsn(term)
            gamma = 1 + (term1*(term2+term3))
            #gamma = 1+(((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))*np.exp(-1*(term**2))*(1j+sp.special.erfi(term)))
            gamma = ((gamma-1)*np.exp(1j*theta)) + 1
            result = np.hstack((np.real(gamma),np.imag(gamma)))
            return result

    return s11_model

def fit_s11(freqs,gamma_data, \
            model = 'rotated', \
            f0_guess = None, \
            g_int_guess = 0.3E6, \
            g_ext_guess = 1.3E6, \
            theta_guess = 0.0, \
            sigma_guess = 1.0E6, \
            bounds = True, \
            tolerance = 1.0E-15, \
            f0_scale = 1.0E9, \
            damping_scale = 1.0E6, \
            theta_scale = 1.0, \
            n_evals = 1E7):
    
    data = np.hstack((np.real(gamma_data),np.imag(gamma_data)))

    if isinstance(f0_guess,type(None)):
        f0_guess = freqs[round((len(freqs)-1)/2)]

    if bounds:
        q_bound = freqs[-1]-freqs[0]
        f0_low = freqs[0]
        f0_high = freqs[-1]
        g_int_low = 0.05E6
        g_int_high = 0.3E6
        g_ext_low = 0
        g_ext_high = q_bound
        theta_low = -np.pi
        theta_high = np.pi
        sigma_low = 0
        sigma_high = q_bound
    else:
        f0_low = 0
        f0_high = np.inf
        g_int_low = 0
        g_int_high = np.inf
        g_ext_low = 0
        g_ext_high = np.inf
        theta_low = -np.pi
        theta_high = np.pi
        sigma_low = 0
        sigma_high = np.inf
        
    if model == 'ideal':
        param_guess = [f0_guess, g_int_guess, g_ext_guess]
        scales = [f0_scale,damping_scale,damping_scale]
        low_bounds = [f0_low, g_int_low, g_ext_low]
        high_bounds = [f0_high, g_int_high, g_ext_high]
        
    elif model == 'rotated':
        param_guess = [f0_guess, g_int_guess, g_ext_guess, theta_guess]
        scales = [f0_scale,damping_scale,damping_scale,theta_scale]
        low_bounds = [f0_low, g_int_low, g_ext_low, theta_low]
        high_bounds = [f0_high, g_int_high, g_ext_high, theta_high]
        
    elif model == 'noisy':
        param_guess = [f0_guess, g_int_guess, g_ext_guess, theta_guess, sigma_guess]
        scales = [f0_scale,damping_scale,damping_scale,theta_scale,damping_scale]
        low_bounds = [f0_low, g_int_low, g_ext_low, theta_low, sigma_low]
        high_bounds = [f0_high, g_int_high, g_ext_high, theta_high, sigma_high]
        
    bounds_tuple = (low_bounds,high_bounds)
    
    popt,pcov = opt.curve_fit(make_s11(model), \
                              freqs, \
                              data, \
                              p0=param_guess, \
                              bounds=bounds_tuple, \
                              ftol=tolerance, \
                              xtol=tolerance, \
                              gtol=tolerance, \
                              x_scale = scales, \
                              max_nfev=n_evals)
    perr = np.sqrt(np.diag(pcov))
    return popt,perr
    
def get_s11_fit(f, params, \
                model='rotated', \
                return_str = 'complex'):
    # model = string indicating model to use
    # return_str = string:
    #               - 'xy' for real and imag
    #               - 'complex' for x+iy
    if model == 'ideal':
            f0, g_int, g_ext = params
            df = f-f0
            numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
            denominator = (4*(df**2))+((g_int + g_ext)**2)
            gamma = numerator/denominator
        
    elif model == 'rotated':
            f0, g_int, g_ext, theta = params
            df = f-f0
            numerator = (4*(df**2))+((g_int**2)-(g_ext**2))+(4j*g_ext*df)
            denominator = (4*(df**2))+((g_int + g_ext)**2)
            gamma = numerator/denominator
            gamma = ((gamma-1)*np.exp(1j*theta)) + 1
        
    elif model == 'noisy':
            f0, g_int, g_ext, theta, sigma = params
            f = f.astype(np.float64)
            df = f-f0
            term = (df-(1j*((g_int+g_ext)/2)))/(sigma*np.sqrt(2))
            term = term.astype(np.complex128)
            term1 = ((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))
            term2 = 1j*np.exp(-1*(term**2))
            term3 = (2/np.sqrt(np.pi))*sp.special.dawsn(term)
            gamma = 1 + (term1*(term2+term3))
            #gamma = 1+(((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))*np.exp(-1*(term**2))*(1j+sp.special.erfi(term)))
            gamma = ((gamma-1)*np.exp(1j*theta)) + 1

    if return_str == 'xy':
        return np.real(gamma),np.imag(gamma)
    elif return_str == 'complex':
        return gamma

def make_s11_duffing(f0, \
                     #g_int, \
                     #g_ext, \
                     theta, \
                     #sigma, \
                     imag_cutoff=1E-20, \
                     sigma_range = 4.0, \
                     bistability_method='hysteresis',\
                     power_units = 'W'):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    def s11_duffing(f,duff,Pin,sigma,g_int,g_ext):
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
        result = np.hstack((np.real(avg_gamma),np.imag(avg_gamma)))
        return result
    return s11_duffing

def fit_s11_duffing(freqs,\
                    gamma_data, \
                    f0, \
                    #g_int, \
                    #g_ext, \
                    theta, \
                    #sigma, \
                    duff_guess = 1.0E5, \
                    Pin_guess = None, \
                    imag_cutoff=1E-20, \
                    sigma_range = 4.0, \
                    bistability_method='hysteresis', \
                    power_units = 'W', \
                    bounds = True, \
                    tolerance = 1.0E-15, \
                    duff_scale = 0.1E5, \
                    n_evals = 1E4, \
                    return_fit = True):
    
    data = np.hstack((np.real(gamma_data),np.imag(gamma_data)))
    
    if power_units == 'dbm':
        if isinstance(Pin_guess,type(None)):
            Pin_guess = -135.0
        power_scale = 0.5
        P_low = -150.0
        P_high = -120.0
    else:
        if isinstance(Pin_guess,type(None)):
            Pin_guess = 3.0E-17
        power_scale = 1.0E-19
        P_low = 5.0E-19
        P_high = 1.0E-15
    
    if bounds:
        q_bound_low = 0.05E6
        q_bound_high = 5.0E6
        low_bounds = [-1.0E6,P_low,q_bound_low,q_bound_low,q_bound_low]
        high_bounds = [1.0E6,P_high,q_bound_high,q_bound_high,q_bound_high]
    else:
        low_bounds = [-np.inf,-np.inf]
        high_bounds = [np.inf,np.inf]
    bounds_tuple = (low_bounds,high_bounds)
    
    scales = [duff_scale,power_scale,0.1E6,0.05E6,0.1E6]
    param_guess = [duff_guess,Pin_guess,1.0E6,0.1E6,1.0E6]
    print(power_units)
    popt,pcov = opt.curve_fit(make_s11_duffing(f0, \
                                               #g_int, \
                                               #g_ext, \
                                               theta, \
                                               #sigma, \
                                               imag_cutoff=imag_cutoff, \
                                               sigma_range = sigma_range, \
                                               bistability_method=bistability_method,\
                                               power_units = power_units), \
                              freqs, \
                              data, \
                              p0=param_guess, \
                              bounds=bounds_tuple, \
                              ftol=tolerance, \
                              xtol=tolerance, \
                              gtol=tolerance, \
                              x_scale = scales, \
                              max_nfev=n_evals, \
                              )
    perr = np.sqrt(np.diag(pcov))
    if return_fit:
        fit_result = make_s11_duffing(f0, \
                                      #g_int, \
                                      #g_ext, \
                                      theta, \
                                      #sigma, \
                                      imag_cutoff=imag_cutoff, \
                                      sigma_range = sigma_range, \
                                      bistability_method=bistability_method,\
                                      power_units = power_units)(freqs,popt[0],popt[1],popt[2],popt[3],popt[4])
        real_part = fit_result[0:int(len(fit_result)/2)]
        im_part = fit_result[int(len(fit_result)/2):]
        gamma_fit = real_part + (1.0j*im_part)
        return popt,perr,gamma_fit
    else:
        return popt,perr

def get_s11_duffing_fit(f,\
                        f0, \
                        g_int, \
                        g_ext, \
                        theta, \
                        sigma, \
                        duff, \
                        Pin, \
                        imag_cutoff=1E-20, \
                        sigma_range = 4.0, \
                        bistability_method='hysteresis', \
                        power_units = 'dBm'):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    if power_units == 'dBm':
        Pin = 10**((Pin-30.0)/10.0)
    f_step = f[1]-f[0] #ASSUMES UNIFORM SPACING
    ind_range = int(round((sigma*sigma_range)/f_step))
    start_arr = f[0]+(f_step*np.linspace(-1*ind_range,-1,ind_range))
    end_arr = f[-1]+(f_step*np.linspace(1,ind_range,ind_range))
    longer_f = np.hstack((start_arr,f,end_arr))
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
    avg_gamma = ((avg_gamma-1)*np.exp(1j*theta)) + 1
    result = np.hstack((np.real(avg_gamma),np.imag(avg_gamma)))
    return result
