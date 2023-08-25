# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:50:12 2020

@author: Ben
"""

import numpy as np

def get_bin_centers(bin_edges):
    bin_centers = bin_edges + (np.roll(bin_edges,-1)-bin_edges)/2.0
    return bin_centers[:-1]

def get_hist(samples, n_bins, density = True, weights=None):
    hist, bin_edges = np.histogram(samples, bins=n_bins, density=density, weights=weights)
    bin_centers = get_bin_centers(bin_edges)
    return hist, bin_centers

def get_hist2d(x_samples, y_samples, n_bins, density = True, return_edges = True):
    # n_bins = int, [int,int], or [array,array] see numpy code
    hist2d, x_edges, y_edges = np.histogram2d(x_samples, y_samples, bins=n_bins,density=density)
    if return_edges:
        return hist2d, x_edges, y_edges
    else:
        x_centers = get_bin_centers(x_edges)
        y_centers = get_bin_centers(y_edges)
        return hist2d, x_centers, y_centers
    
def gaussian_pdf(x,m,s):
    return (1/np.sqrt(2*np.pi*(s**2)))*np.exp((-(x-m)**2)/(2*(s**2)))
