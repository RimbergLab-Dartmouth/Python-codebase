# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:59:20 2019

@author: Ben
"""
import numpy as np


def duffing_response(f, f0, g_int, g_ext, duff, Pin, \
                     imag_cutoff=1E-20,\
                     bistability_method = 'hysteresis'):
    # bistability_method = 'hysteresis' or 'boltzmann'
    t0=time.time()
    roots_time = 0
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    response = np.zeros_like(f)
    prev_root = None
    for ii in range(len(f)):
        freq = f[ii]
        coeffs = np.array([1.0, \
                           (2*(f0-freq))/duff, \
                           (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/((duff**2)), \
                           (-g_ext*Pin)/(2*np.pi*h*freq*(duff**2))])
        t2 = time.time()
        roots = np.roots(coeffs)
        t3 = time.time()
        roots_time += t3-t2
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
    t1 = time.time()
    #print('duffing response time: '+str(t1-t0)+' seconds')
    #print('roots time: '+str(roots_time)+' seconds')
    #print('total - roots time = '+str(t1-t0-roots_time)+' seconds')
    return np.array(response)

def s11_duffing_theory(f, f0, g_int, g_ext, sigma, duff, Pin, \
                       imag_cutoff=1E-20, \
                       x_samples=50, \
                       sigma_range = 3.0, \
                       bistability_method='hysteresis'):
    df_arr = np.zeros((len(f),x_samples))
    x_axis = np.linspace(-1*sigma_range*sigma,sigma_range*sigma,x_samples)
    dx = x_axis[1]-x_axis[0]
    gauss = (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-(x_axis)**2)/(2*(sigma**2)))
    for ii in range(x_samples):
        x = x_axis[ii]
        response = duffing_response(f, f0+x, g_int, g_ext, duff, Pin, imag_cutoff,bistability_method=bistability_method)
        df_arr[:,ii] = f-f0-x-(duff*response)
    gamma_integrand = (df_arr-((1j*(g_int-g_ext))/2))/(df_arr-((1j*(g_int+g_ext))/2))
    gamma = dx*np.sum(gauss*gamma_integrand,axis=1)
    return gamma


def s11_duffing_theory2(f, f0, g_int, g_ext, sigma, duff, Pin, \
                       imag_cutoff=1E-20, \
                       x_samples=50, \
                       sigma_range = 3.0, \
                       bistability_method='hysteresis'):
    gamma_arr = np.zeros((len(f),x_samples),dtype=np.complex128)
    samples = np.random.normal(loc=0.0,scale=sigma,size=x_samples)
    for ii in range(x_samples):
        x = samples[ii]
        response = duffing_response(f, f0+x, g_int, g_ext, duff, Pin, imag_cutoff,bistability_method=bistability_method)
        df = f-f0-x-(duff*response)
        gamma_arr[:,ii] = (df-((1j*(g_int-g_ext))/2))/(df-((1j*(g_int+g_ext))/2))
    gamma = np.mean(gamma_arr,axis=1)
    return gamma

# BEST IS THE ONE BELOW

def s11_duffing_theory3(f, f0, g_int, g_ext, sigma, duff, Pin, \
                       imag_cutoff=1E-20, \
                       x_samples = 100, \
                       sigma_range = 3.0, \
                       bistability_method='hysteresis'):
    f_step = f[1]-f[0] #ASSUMES UNIFORM SPACING
    ind_range = int(round((sigma*sigma_range)/f_step))
    start_arr = f[0]+(f_step*np.linspace(-1*ind_range,-1,ind_range))
    end_arr = f[-1]+(f_step*np.linspace(1,ind_range,ind_range))
    longer_f = np.hstack((start_arr,f,end_arr))
    response = duffing_response(longer_f, f0, g_int, g_ext, duff, Pin, imag_cutoff,bistability_method=bistability_method)
    df = longer_f-f0-(duff*response)
    gamma = (df-((1j*(g_int-g_ext))/2))/(df-((1j*(g_int+g_ext))/2))
    x_axis = np.linspace(-f_step*ind_range,f_step*ind_range,(2*ind_range+1))
    gauss = (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp((-(x_axis)**2)/(2*(sigma**2)))
    avg_gamma = np.zeros(len(f),dtype=np.complex128)
    for ii in range(len(f)):
        avg_gamma[ii] = f_step*np.sum(gamma[ii:ii+(2*ind_range)+1]*gauss)
    return avg_gamma


def duffing_response(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-20):
    h = 6.626*(10**-34)
    response = []
    prev_root = None
    bistability_flag = 0
    for freq in f:
        coeffs = np.array([1.0, \
                           (f0-freq)/duff, \
                           (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/(4*(duff**2)), \
                           (-g_ext*Pin)/(8*np.pi*h*freq*(duff**2))])
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
    return np.array(response), bistability_flag

def s11_duffing(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-20):
    response, bistability_flag = duffing_response(f,f0,g_int,g_ext,duff,Pin,imag_cutoff)
    detuning = f-f0
    numerator = (1j*(detuning-(2*duff*response)))+((g_int-g_ext)/2)
    denominator = (1j*(detuning-(2*duff*response)))+((g_int+g_ext)/2)
    gamma = numerator/denominator
    return gamma, bistability_flag


def avg_duffing_response(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-30):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    response = []
    for freq in f:
        coeffs = np.array([1.0, \
                           (f0-freq)/duff, \
                           (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/(4*(duff**2)), \
                           (-g_ext*Pin)/(8*np.pi*h*freq*(duff**2))])
        roots = np.roots(coeffs)
#        print(roots)
        im_part = np.imag(roots)
        real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
#        print('im part: ')
#        print(im_part)
#        print('real roots:')
#        print(real_roots)
#        input()
        if len(real_roots)>1:
            min_root = np.min(real_roots)
            max_root = np.max(real_roots)
            min_energy = (h*f0*min_root)+(h*duff*(min_root**2))
            max_energy = (h*f0*max_root)+(h*duff*(max_root**2))
            min_weight = np.exp((-1*min_energy)/kT)
            max_weight = np.exp((-1*max_energy)/kT)
            root = ((min_root*min_weight)+(max_root*max_weight))/(min_weight+max_weight)
        else:
            root = real_roots[0]
        response.append(root)
    return np.array(response)

def s11_avg_duffing(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-30):
    response = avg_duffing_response(f,f0,g_int,g_ext,duff,Pin,imag_cutoff)
    detuning = f-f0
    numerator = (1j*(detuning-(2*duff*response)))+((g_int-g_ext)/2)
    denominator = (1j*(detuning-(2*duff*response)))+((g_int+g_ext)/2)
    gamma = numerator/denominator
    return gamma, bistability_flag


def s11_everything(f,f0,g_int,g_ext,sigma,theta,kerr,Pin):
    #Pin = np.power(10,(Pdbm/10)-3)
    response = avg_duffing_response(f,f0,g_int,g_ext,kerr,Pin)
    df = f-f0-(kerr*response)
    term = (df-(1j*((g_int+g_ext)/2)))/(sigma*np.sqrt(2))
    gamma = 1+(((1j*np.pi*g_ext)/(np.sqrt(2*np.pi*(sigma**2))))*np.exp(-1*(term**2))*(1j+sp.special.erfi(term)))
    rotated_gamma = ((gamma-1)*np.exp(1j*theta)) + 1
    result = np.hstack((np.real(rotated_gamma),np.imag(rotated_gamma)))
    return result

def fit_everything(freqs,data, \
                        f0_guess = None, \
                        g_int_guess = 0.3E6, \
                        g_ext_guess = 1.3E6, \
                        sigma_guess = 0.5E6, \
                        theta_guess = 0.0, \
                        kerr_guess = 0.5E6, \
                        Pin_guess = 1E-16, \
                        bounds = True):
    if not f0_guess:
        f0_guess = freqs[round(len(freqs)/2)]
    param_guess = [f0_guess, g_int_guess, g_ext_guess, sigma_guess, theta_guess, kerr_guess, Pin_guess]
    if bounds:
        q_bound = freqs[-1]-freqs[0]
        low_bounds = [freqs[0],0.1E6,0,0,-np.pi,-5.0E6, 1E-18]
        high_bounds = [freqs[-1],q_bound,q_bound,q_bound,np.pi, 5.0E6, 1E-14]
        bounds_tuple = (low_bounds,high_bounds)
    else:
        low_bounds = [-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf]
        high_bounds = [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]
        bounds_tuple = (low_bounds,high_bounds)
    popt,pcov = opt.curve_fit(s11_everything, \
                              freqs, \
                              data, \
                              p0=param_guess, \
                              bounds=bounds_tuple)
    perr = np.sqrt(np.diag(pcov))
    return popt,perr

def duffing_empirical(f, f0, g_int, g_ext, theta0, kerr, Pin):
    h = 6.626*(10**-34)
    pass

def duffing_discriminant(f,f0,g_int,g_ext,duff,Pin):
    h = 6.626*(10**-34)
    discriminant_list = []
    for freq in f:
        coeffs = np.array([1.0, \
                           (f0-freq)/duff, \
                           (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/(4*(duff**2)), \
                           (-g_ext*Pin)/(8*np.pi*h*freq*(duff**2))])
        a = coeffs[0]
        b = coeffs[1]
        c = coeffs[2]
        d = coeffs[3]
        discriminant = 18*a*b*c*d - \
                        4*(b**3)*d + \
                        (b**2)*(c**2) - \
                        4*a*(c**3) - \
                        27*(a**2)*(d**2)
        discriminant_list.append(discriminant)
    return np.array(discriminant_list)

def duffing_response(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-20):
    h = 6.626*(10**-34)
    response = []
    prev_root = None
    bistability_flag = 0
    for freq in f:
        coeffs = np.array([1.0, \
                           (f0-freq)/duff, \
                           (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/(4*(duff**2)), \
                           (-g_ext*Pin)/(8*np.pi*h*freq*(duff**2))])
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
                #print('MORE THAN ONE REAL ROOT!')
                bistability_flag = 1
            if prev_root:
                root = real_roots[np.argmin(np.abs(real_roots-prev_root))]
            else:
                root = real_roots[np.argmin(real_roots)]
        else:
            root = real_roots[0]
        response.append(root)
        prev_root = root
    return np.array(response), bistability_flag

def avg_duffing_response(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-30):
    h = 6.626*(10**-34)
    kT = 4.14*(10**-25)
    response = []
    bistability_flag = 0
    for freq in f:
        coeffs = np.array([1.0, \
                           (f0-freq)/duff, \
                           (((freq-f0)**2)+(((g_int+g_ext)**2)/4))/(4*(duff**2)), \
                           (-g_ext*Pin)/(8*np.pi*h*freq*(duff**2))])
        roots = np.roots(coeffs)
#        print(roots)
        im_part = np.imag(roots)
        real_roots = np.real(roots[np.where(np.abs(im_part)<imag_cutoff)])
#        print('im part: ')
#        print(im_part)
#        print('real roots:')
#        print(real_roots)
#        input()
        if len(real_roots)>1:
            min_root = np.min(real_roots)
            max_root = np.max(real_roots)
            min_energy = (h*f0*min_root)+(h*duff*(min_root**2))
            max_energy = (h*f0*max_root)+(h*duff*(max_root**2))
            min_weight = np.exp((-1*min_energy)/kT)
            max_weight = np.exp((-1*max_energy)/kT)
            root = ((min_root*min_weight)+(max_root*max_weight))/(min_weight+max_weight)
            if not bistability_flag:
                bistability_flag = 1
        else:
            root = real_roots[0]
        response.append(root)
    return np.array(response), bistability_flag

def s11_avg_duffing(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-30):
    response, bistability_flag = avg_duffing_response(f,f0,g_int,g_ext,duff,Pin,imag_cutoff)
    detuning = f-f0
    numerator = (1j*(detuning-(2*duff*response)))+((g_int-g_ext)/2)
    denominator = (1j*(detuning-(2*duff*response)))+((g_int+g_ext)/2)
    gamma = numerator/denominator
    return gamma, bistability_flag

def s11_duffing(f, f0, g_int, g_ext, duff, Pin, imag_cutoff=1E-30):
    response, bistability_flag = duffing_response(f,f0,g_int,g_ext,duff,Pin,imag_cutoff)
    detuning = f-f0
    numerator = (1j*(detuning-(2*duff*response)))+((g_int-g_ext)/2)
    denominator = (1j*(detuning-(2*duff*response)))+((g_int+g_ext)/2)
    gamma = numerator/denominator
    return gamma, bistability_flag