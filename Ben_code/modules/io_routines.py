# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 11:24:11 2018

@author: Ben
"""
import os
import numpy as np
import ast

def write_array(array, filename):
    # store array shape + flattened data
    shape_str = str(np.shape(array))
    data = np.ravel(array)
    data = [str(item) for item in data]
    data = ','.join(data)
    array_str = shape_str+'\n'+data
    try_remove(filename)
    f = open(filename,'w')
    f.write(array_str)
    f.close()
    
def read_array(filename):
    # read an array stored as shape + flattened data
    f = open(filename,'r')
    array_str = f.read()
    f.close()
    array_str = array_str.split('\n')
    shape = ast.literal_eval(array_str[0])
    data = array_str[1].split(',')
    data = [float(item) for item in data]
    array = np.reshape(data,shape)
    return array

def try_remove(filename):
    # remove a file if it already exists, otherwise do nothing
    try:
        os.remove(filename)
    except OSError:
        pass

def parse_csv(filename):
    # parse a file of comma separated values, where the values are floats
    f = open(filename,'r')
    data = f.read()
    f.close()
    data = data.split(',')
    data = [float(item) for item in data]
    return data

def parse_csv_array(filename):
    # same as parse csv, but it returns an array rather than a list
    return np.array(parse_csv(filename))

def parse_csv_list(filename):
    # parse a file containing a list of lists
    # first level lists separated by new line \n
    # second level list is csv
    f = open(filename,'r')
    data = f.read()
    f.close()
    data = data.split('\n')
    for ii in range(len(data)):
        data[ii] = data[ii].split(',')
        data[ii] = [float(item) for item in data[ii]]
    return data

def get_float(filename):
    # retrieve a file in which a single float is stored as a string
    f = open(filename,'r')
    data = f.read()
    f.close()
    return float(data)


    
def write_csv(data, filename):
    # takes a list of floats and writes it to a csv at filename
    data = [str(item) for item in data]
    data = ','.join(data)
    try_remove(filename)
    f = open(filename,'w')
    f.write(data)
    f.close()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
