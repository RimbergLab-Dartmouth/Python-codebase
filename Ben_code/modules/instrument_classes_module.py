# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:36:20 2018

@author: Ben
"""

import numpy as np
import statistics
import struct
import visa
import nidaqmx
import tkinter as tk
import time
import atsapi as ats
import ctypes
import sys
import math
import iq_routines as iq

class gpib_instrument:

    def __init__(self, addr):
        addr_str = 'GPIB0::'+str(addr)+'::INSTR'
        self.instr = visa.ResourceManager().open_resource(addr_str)
    
    def write(self, message, return_output = 0):
        # message = str
        # return_output = 1 to return the instrument's response
        if return_output:
            return self.instr.write(message)
        else:
            self.instr.write(message)
        
    def query(self, message):
        # message = str
        return self.instr.query(message)
    
    def write_raw(self, message, return_output = 0):
        # message = bytes
        if return_output:
            return self.instr.write_raw(message)
        else:
            self.instr.write_raw(message)

    def read_raw(self):
        # message = str
        return self.instr.read_raw()

    def set_timeout(self,timeout):
        # sets timeout (in milliseconds)
        # timeout = float
        self.instr.timeout = timeout

    def wai(self,):
        self.write('*wai')
        
    def opc(self):
        self.query('*opc?')
        
    def rst(self):
        self.write('*rst')
        
    def idn(self):
        return self.query('*idn?')


"""
*******************************************************************************
*******************************************************************************
"""

class keysight_n5183b(gpib_instrument):
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)

    def set_frequency(self, freq):
        #freq = float
        message = ':freq '+str(freq)
        self.write(message)
        
    def set_power(self, power, units = 'dbm'):
        # power = float
        # units = str
        #   options: 'dbm', 'mv', 'dBuV', 'dBuVemf', 'uV', 'mVemf', 'uVemf'
        message = ':pow '+str(power)+units
        self.write(message)
        
    def set_phase(self, phase, units = 'rad'):
        # phase = float
        # units = str: 'rad' or 'deg'
        message = ':phas '+str(phase)+units
        self.write(message)
        
    def toggle_output(self, state):
        # turns RF output on or off
        # state = 0,1 (off,on)
        message = ':outp '+str(state)
        self.write(message)
        
    def toggle_modulation(self, state):
        # turns modulation on or off
        # state = 0,1 (off,on)
        message = ':outp:mod '+str(state)
        self.write(message)
        
    def toggle_pulse_mode(self, state):
        # turns pulse mode on or off
        # state = 0,1 (off,on)
        message = ':pulm:stat '+str(state)
        self.write(message)

    def toggle_alc(self, state):
        # turns on and off automatic leveling control
        #   - useful for ultra-narrow-width (UNW) pulse generation
        # state = 0,1 (off,on)
        message = ':pow:alc '+str(state)
        self.write(message)
        
    def set_pulse_source(self, source):
        # sets the source for the pulse modulation tone
        # source = str
        #   options:
        #       'ext'   -external pulse modulates output
        #       'trig'  -internal source, triggered
        #       'frun'  -internal source, free run
        #       'ado'   -internal source, see manual
        #       'doub'  -internal source, see manual
        #       'gate'  -internal source, see manual
        #       'ptr'   -internal source, see manual
        if source == 'ext':
            message = 'pulm:sour '+source
        else:
            message = 'pulm:sour:int '+source
                
        self.write(message)

    def set_pulse_delay(self, delay, units='s'):
        # sets pulse delay
        # delay = float
        # units = str: 's','us','ns'
        message = ':pulm:int:del '+str(delay)+units
        self.write(message)
        
    def set_pulse_width(self, width, units='s'):
        # sets pulse delay
        # width = float
        # units = str: 's','us','ns'
        message = ':pulm:int:pwid '+str(width)+units
        self.write(message)
        
    def set_pulse_period(self, period, units = 's'):
        # sets pulse period
        # period = float
        message = ':pulm:int:per '+str(period)+units
        self.write(message)
        
    def route_trigger_out(self, signal_str, ind=1):
        # routes signal to output on the trigger (#ind) port
        # ind = 1,2 for trigger 1,2 port
        # signal_str    = SWEep (sweep trigger)
        #               = SETTled (source settled)
        #               = PVIDeo (pulse video)
        #               = PSYNc (pulse sync)
        #               = PULSe (pulse or pulse sync?)
        #               see manual for other options
        message = ':rout:trig'+str(ind)+':outp '+signal_str
        self.write(message)

"""
*******************************************************************************
*******************************************************************************
"""    
    
class agilent_e8257c(gpib_instrument):
    # SIGNAL GENERATOR
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)

    def toggle_output(self, state):
        # turns output off or on
        # state = 0,1 (off,on)
        message = ':outp '+str(state)
        self.write(message)
        
    def toggle_modulation(self, state):
        # turns modulation off or on
        # state = 0,1 (off,on)
        message = ':outp:mod '+str(state)
        self.write(message)
    
    def set_frequency(self, freq, units = 'Hz'):
        # set the frequency
        # freq = float , scientific notation OK
        # units = str: 'Hz', 'kHz', 'MHz', 'GHz' all OK
        message = ':freq '+str(freq)+units
        self.write(message)
        
    def set_phase(self, phase, units = 'rad'):
        # phase = float
        # units = str: 'rad' or 'deg'
        message = ':phas '+str(phase)+units
        self.write(message)
        
    def toggle_alc(self, state):
        # turns automatic leveling control off,on
        # state = 0,1 (off,on)
        message = ':pow:alc '+str(state)
        self.write(message)
        
    def set_power(self, power, units = 'dbm'):
        # power = float
        # units = str
        #   options: 'dbm', 'mv', 'dBuV', 'dBuVemf', 'uV', 'mVemf', 'uVemf'
        message = ':pow '+str(power)+units
        self.write(message)
        
    def toggle_pulse_mode(self, state):
        # turns pulse mode on or off
        # state = 0,1 (off,on)
        message = ':pulm:stat '+str(state)
        self.write(message)

    def set_pulse_source(self, source):
        # sets the source for the pulse modulation tone
        # source = str
        #   options:
        #       'ext'   -external pulse modulates output
        #       'squ'   -internal source, square
        #       'trig'  -internal source, triggered
        #       'frun'  -internal source, free run
        #       'doub'  -internal source, see manual
        #       'gate'  -internal source, see manual
        if source == 'ext':
            message = 'pulm:sour '+source
        else:
            message = 'pulm:sour:int '+source
                
        self.write(message)

    def set_pulse_delay(self, delay, units='s'):
        # sets pulse delay
        # delay = float
        # units = str: 's','us','ns'
        message = ':pulm:int:del '+str(delay)+units
        self.write(message)
        
    def set_pulse_width(self, width, units='s'):
        # sets pulse delay
        # width = float
        # units = str: 's','us','ns'
        message = ':pulm:int:pwid '+str(width)+units
        self.write(message)


"""
*******************************************************************************
*******************************************************************************
"""    

class hp_83711b(gpib_instrument):
    # SIGNAL GENERATOR
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)

    def set_frequency(self, freq, units = 'Hz'):
        # set the frequency
        # freq = float , scientific notation OK
        # units = str: 'Hz', 'kHz', 'MHz', 'GHz' all OK
        message = 'freq '+str(freq)+units
        self.write(message)

    def set_power(self, power, units = 'dBm'):
        # set the power
        # power = float
        # units = str: 'dBm', 'mV', 'V', etc.
        message = 'pow '+str(power)+units
        self.write(message)

    def toggle_output(self, state):
        # turns output off or on
        # state = 0,1 (off,on)
        message = 'outp '+str(state)
        self.write(message)


"""
*******************************************************************************
*******************************************************************************
"""    

class hp_8648c(gpib_instrument):
    # SIGNAL GENERATOR
    
    def __init__(self, addr):
        super().__init__(addr)

    def set_frequency(self, freq, units='Hz'):
        message = ':freq ' + str(freq) + units
        self.write(message)

    def toggle_output(self, state):
        message = 'outp ' + str(state)
        self.write(message)

    def set_power(self, power, units='dBm'):
        message = 'pow ' + str(power) + units
        self.write(message)

"""
*******************************************************************************
*******************************************************************************
"""    

class hp_34401a(gpib_instrument):
    # MULTIMETER
    
    def __init__(self, addr):
        super().__init__(addr)

    def get_voltage(self):
        message = 'meas:volt:dc?'
        return float(self.query(message))
    
    def get_current(self):
        message = 'meas:curr:dc?'
        return float(self.query(message))

"""
*******************************************************************************
*******************************************************************************
"""    

class agilent_e3634a(gpib_instrument):
    # DC POWER SUPPLY
    
    def __init__(self, addr):
        super().__init__(addr)

    def toggle_output(self, state):
        message = 'outp ' + str(state)
        self.write(message)

    def apply(self, voltage, current, v_units='V', i_units='A'):
        message = 'appl ' + str(voltage) + v_units + ', ' + str(current) + i_units
        self.write(message)

    def set_voltage(self, voltage, units='V'):
        message = 'volt ' + str(voltage) + units
        self.write(message)

    def set_current(self, current, units='A'):
        message = 'curr ' + str(current) + units
        self.write(message)

    def measure_voltage(self):
        message = 'meas:volt?'
        return self.query(message)

    def measure_current(self):
        message = 'meas:curr?'
        return self.query(message)

"""
*******************************************************************************
*******************************************************************************
"""    

class hp_6612c(gpib_instrument):
    # DC POWER SUPPLY
    
    def __init__(self, addr):
        super().__init__(addr)

    def toggle_output(self, state):
        message = 'outp ' + str(state)
        self.write(message)


"""
*******************************************************************************
*******************************************************************************
"""    

class keysight_n9000b(gpib_instrument):
    # SPECTRUM ANALYZER
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)
        
    def set_mode(self, mode_str):
        # sets analysis mode
        # mode_str = str
        #   options:
        #       'sa'    - spectrum analysis
        #       'basic' - basic I/Q analysis
        message = 'inst:sel '+mode_str
        self.write(message)
        
    def select_screen(self, screen_str):
        # screen_str = name of screen surrounded by double quotes
        message = ':inst:scr:sel '+screen_str
        self.write(message)
        
    def read_trace(self,convert=True):
        # performs a sweep and returns data
        # format: freq_str, pow_str, freq_str, pow_str, ...
        # I omit the new line character at the end of the trace
        message = 'read:san?'
        sa_trace = self.query(message)[:-1]
        if convert:
            sa_trace = sa_trace.split(',')
            freqs = np.array([float(sa_trace[2*ii]) for ii in range(len(sa_trace)//2)])
            pows = np.array([float(sa_trace[2*ii+1]) for ii in range(len(sa_trace)//2)])
            return freqs,pows
        else:
            return sa_trace
    
    def fetch_trace(self):
        # retrieves trace already taken
        # format: freq_str, pow_str, freq_str, pow_str, ...
        # I omit the new line character at the end of the trace
        message = 'fetch:san?'
        return self.query(message)[:-1]
    
    def get_all_peaks(self):
        # get the peaks of a trace in x,y pairs
        # I exclude the final new line character \n
        message = 'trac:math:peak?'
        return self.query(message)[:-1]
    
    def get_peaks(self, threshold, excursion):
        # NOTE: RETURN FORMAT:
        #       num_peaks, peak1_power, peak1_freq, peak2_power, ...
        # get all peaks above a certain threshold & excursion
        # threshold = float (minimum power level in dBm)
        # excursion = float (rise and fall in dBm)
        message = ':calc:data:peak? '+str(threshold)+','+str(excursion)
        return self.query(message)[:-1]
    
    def toggle_automatic_resolution_bandwidth(self, state):
        # turn automatic coupling of resolution bw off or on
        # state = 0,1 (off,on)
        message = ':band:auto ' + str(state)
        self.write(message)
        
    def set_resolution_bandwidth(self, bandwidth, units='Hz'):
        # set the resolution bandwidth
        # bandwidth = float: scientific notation OK
        # units = str
        message = ':band ' + str(bandwidth) + units
        self.write(message)
        
    def get_resolution_bandwidth(self):
        # query the resolution bandwidth (in Hz)
        message = ':band?'
        return float(self.query(message))
    
    def set_video_bandwidth(self, bandwidth, units='Hz'):
        # set the video bandwidth
        # bandwidth = float: scientific notation OK
        # units = str
        message = ':band:vid ' + str(bandwidth) + units
        self.write(message)

    def get_video_bandwidth(self):
        # query the video bandwidth (in Hz)
        message = ':band:vid?'
        return float(self.query(message))

    def toggle_automatic_video_bandwidth(self, state):
        # turn automatic coupling of video bw off or on
        # state = 0,1 (off,on)
        message = ':band:vid:auto ' + str(state)
        self.write(message)

    def set_frequency_center(self, frequency, units='Hz'):
        # frequency = float: scientific notation OK
        message = ':freq:cent ' + str(frequency) + units
        self.write(message)

    def set_frequency_span(self, frequency, units='Hz'):
        # frequency = float: scientific notation OK
        message = ':freq:span ' + str(frequency) + units
        self.write(message)

    def set_frequency_start(self, frequency, units='Hz'):
        # frequency = float: scientific notation OK
        message = ':freq:star ' + str(frequency) + units
        self.write(message)

    def set_frequency_stop(self, frequency, units='Hz'):
        # frequency = float: scientific notation OK
        message = ':freq:stop ' + str(frequency) + units
        self.write(message)

    def set_freqs(self, freq1, freq2, interval='span'):
        if interval == 'range':
            self.set_frequency_start(freq1)
            self.set_frequency_stop(freq2)
        else:
            if interval == 'span':
                self.set_frequency_center(freq1)
                self.set_frequency_span(freq2)
            else:
                print('ERROR: Invalid Interval Type!')
                
    def set_averaging(self, counts):
        # counts = int
        # note: may initiate or augment a sweep
        message = ':aver:count ' + str(counts)
        self.write(message)
        
    def set_averaging_type(self, type_str):
        # type_str = str
        #   options: 'log', 'rms', 'scal'
        message = ':aver:type ' + type_str
        self.write(message)
        
    def toggle_continuous_sweep(self, state):
        # on: continuous sweep
        # off: single sweep
        message = ':init:cont ' + str(state)
        self.write(message)

    def restart(self):
        # restart current sweep
        message = ':init:rest'
        self.write(message)
        
    def set_sweep_points(self, num_points):
        # sets the number of frequency points in a sweep
        # max = 20,001
        message = ':swe:poin ' + str(num_points)
        self.write(message)
        
        
    def abort(self):
        # abort current sweep
        message = ':abor'
        self.write(message)
        
    def set_trace_type(self, type_str):
        # type_str = str
        #   options:
        #       'WRITe'     - clear/write
        #       'AVERage'   - trace average
        #       'MAXHold'
        #       'MINHold'
        message = 'trac:type '+type_str
        self.write(message)
        
    def set_detector_type(self, type_str):
        # type_str = str
        #   options:
        #       'AVERage'   - good for noise
        #       'NEGative'
        #       'NORMal'    - works simultaneously for pure tones and noise
        #       'POSitive'  - positive peak, good for measuring pure tones
        #       'SAMPle'    - good for noise
        message = ':det:trac '+type_str
        self.write(message)
        
    def trigger(self, wait=True):
        # force trigger
        message = '*trg'
        self.write(message)
        if wait:
            return self.query('*opc?')

    def set_iq_resolution_bandwidth(self, freq, units='Hz'):
        # set the resolution bandwidth for the IQ spectrum
        message = ':spec:band '+str(freq)+units
        self.write(message)
        
    def set_iq_averaging(self, count):
        # set averaging count for IQ spectrum
        message = ':spec:aver:coun '+str(count)
        self.write(message)
        
    def read_iq_waveform(self, return_time_config = True):
        # return iq waveform and time configuration data
        # the waveform and config will be split by a new line '\n'
        message = 'read:wav0?'
        waveform = self.query(message)
        if return_time_config:
            message2 = 'fetch:wav1?'
            time_config = self.query(message2)[:-1]
            waveform += time_config
        return waveform
        



"""
*******************************************************************************
*******************************************************************************
"""    

class agilent_e4404b(gpib_instrument):
    # SPECTRUM ANALYZER
    # ONLY TESTED FOR 4408B, SHOULD BE THE SAME
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)
        
    def abort(self):
        message = ':abor'
        self.write(message)

    def trigger(self):
        message = '*trg'
        self.write(message)

    def toggle_coupling(self, state):
        if state:
            cpl_str = 'all'
        else:
            cpl_str = 'none'
        message = ':coup ' + cpl_str
        self.write(message)

    def toggle_continuous_sweep(self, state):
        message = ':init:cont ' + str(state)
        self.write(message)

    def restart(self):
        message = ':init:rest'
        self.write(message)

    def set_input_coupling(self, coupling_str):
        message = ':inp:coup ' + coupling_str
        self.write(message)

    def toggle_averaging(self, state):
        message = ':aver ' + str(state)
        self.write(message)

    def set_averaging(self, counts):
        message = ':aver:count ' + str(counts)
        self.write(message)

    def set_averaging_type(self, type_str):
        message = ':aver:type ' + type_str
        self.write(message)

    def set_resolution_bandwidth(self, bandwidth, units='Hz'):
        message = ':band ' + str(bandwidth) + units
        self.write(message)

    def get_resolution_bandwidth(self):
        message = ':band?'
        return float(self.query(message))

    def toggle_automatic_resolution_bandwidth(self, state):
        message = ':band:auto ' + str(state)
        self.write(message)

    def set_video_bandwidth(self, bandwidth, units='Hz'):
        message = ':band:vid ' + str(bandwidth) + units
        self.write(message)

    def get_video_bandwidth(self):
        message = ':band:vid?'
        return float(self.query(message))

    def toggle_automatic_video_bandwidth(self, state):
        message = ':band:vid:auto ' + str(state)
        self.write(message)

    def set_detection_type(self, type_str):
        message = ':det ' + type_str
        self.write(message)

    def set_frequency_center(self, frequency, units='Hz'):
        message = ':freq:cent ' + str(frequency) + units
        self.write(message)

    def set_frequency_span(self, frequency, units='Hz'):
        message = ':freq:span ' + str(frequency) + units
        self.write(message)

    def set_frequency_start(self, frequency, units='Hz'):
        message = ':freq:star ' + str(frequency) + units
        self.write(message)

    def set_frequency_stop(self, frequency, units='Hz'):
        message = ':freq:stop ' + str(frequency) + units
        self.write(message)

    def set_freqs(self, freq1, freq2, interval='range', channel=None):
        if interval == 'range':
            self.set_frequency_start(freq1)
            self.set_frequency_stop(freq2)
        else:
            if interval == 'span':
                self.set_frequency_center(freq1)
                self.set_frequency_span(freq2)
            else:
                print('ERROR: Invalid Interval Type!')

    def set_sweep_points(self, num_points):
        message = ':swe:poin ' + str(num_points)
        self.write(message)

    def get_trace_data(self, convert=True):
        message = ':trac? trace1'
        trace = self.query(message)
        if convert:
            trace = trace.split(',')
            trace = [float(ii) for ii in trace]
        return trace

"""
*******************************************************************************
*******************************************************************************
"""    

class agilent_e4408b(gpib_instrument):
    # SPECTRUM ANALYZER
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)

    def abort(self):
        message = ':abor'
        self.write(message)

    def trigger(self):
        message = '*trg'
        self.write(message)

    def toggle_coupling(self, state):
        if state:
            cpl_str = 'all'
        else:
            cpl_str = 'none'
        message = ':coup ' + cpl_str
        self.write(message)

    def toggle_continuous_sweep(self, state):
        message = ':init:cont ' + str(state)
        self.write(message)

    def restart(self):
        message = ':init:rest'
        self.write(message)

    def set_input_coupling(self, coupling_str):
        message = ':inp:coup ' + coupling_str
        self.write(message)

    def toggle_averaging(self, state):
        message = ':aver ' + str(state)
        self.write(message)

    def set_averaging(self, counts):
        message = ':aver:count ' + str(counts)
        self.write(message)

    def set_averaging_type(self, type_str):
        message = ':aver:type ' + type_str
        self.write(message)

    def set_resolution_bandwidth(self, bandwidth, units='Hz'):
        message = ':band ' + str(bandwidth) + units
        self.write(message)

    def get_resolution_bandwidth(self):
        message = ':band?'
        return float(self.query(message))

    def toggle_automatic_resolution_bandwidth(self, state):
        message = ':band:auto ' + str(state)
        self.write(message)

    def set_video_bandwidth(self, bandwidth, units='Hz'):
        message = ':band:vid ' + str(bandwidth) + units
        self.write(message)

    def get_video_bandwidth(self):
        message = ':band:vid?'
        return float(self.query(message))

    def toggle_automatic_video_bandwidth(self, state):
        message = ':band:vid:auto ' + str(state)
        self.write(message)

    def set_detection_type(self, type_str):
        message = ':det ' + type_str
        self.write(message)

    def set_frequency_center(self, frequency, units='Hz'):
        message = ':freq:cent ' + str(frequency) + units
        self.write(message)

    def set_frequency_span(self, frequency, units='Hz'):
        message = ':freq:span ' + str(frequency) + units
        self.write(message)

    def set_frequency_start(self, frequency, units='Hz'):
        message = ':freq:star ' + str(frequency) + units
        self.write(message)

    def set_frequency_stop(self, frequency, units='Hz'):
        message = ':freq:stop ' + str(frequency) + units
        self.write(message)

    def set_freqs(self, freq1, freq2, interval='range', channel=None):
        if interval == 'range':
            self.set_frequency_start(freq1)
            self.set_frequency_stop(freq2)
        else:
            if interval == 'span':
                self.set_frequency_center(freq1)
                self.set_frequency_span(freq2)
            else:
                print('ERROR: Invalid Interval Type!')

    def set_sweep_points(self, num_points):
        message = ':swe:poin ' + str(num_points)
        self.write(message)

    def get_trace_data(self, convert=True):
        message = ':trac? trace1'
        trace = self.query(message)
        if convert:
            trace = trace.split(',')
            trace = [float(ii) for ii in trace]
        return trace

"""
*******************************************************************************
*******************************************************************************
"""    

class agilent_e5071c(gpib_instrument):
    # NETWORK ANALYZER
    
    def __init__(self, addr):
        # inherit all the methods of gpib_instrument class
        super().__init__(addr)
    
    ####################
    # DISPLAY SETTINGS #
    ####################
    
    def allocate_channels(self, alloc_str):
        # splits the display into separate windows
        # channel_str = str
        #   options:
        #       'D1'    - one graph in full window
        #       'D12'   - two graphs split left/right
        #       'D1_2'  - two graphs split top/bottom
        #   see manual for more complicated allocations
        message = ':disp:spl ' + alloc_str
        self.write(message)
        
    def save_channel_state(self, register_str = 'A'):
        # saves state of active channel into the given register
        # register_str = 'A', 'B', 'C', or 'D'
        message = ':mmem:stor:chan ' + register_str
        self.write(message)
        
    def recall_channel_state(self, register_str = 'A'):
        # recalls channel state in the given register into the active channel
        # register_str = 'A', 'B', 'C', or 'D'
        message = ':mmem:load:chan ' + register_str
        self.write(message)
        
    def save_instrument_state(self, register):
        # saves instrument state to register
        # register = int: 1 through 8
        message = ':mmem:stor "D:State0'+str(register)+'.sta"'
        self.write(message)
    
    def recall_instrument_state(self, register):
        # recalls instrument state from register
        # register = int: 1 through 8
        message = ':mmem:load "D:State0'+str(register)+'.sta"'
        self.write(message)

    def set_channel(self,channel):
        # sets active channel
        # channel = int
        message = ':disp:wind'+str(channel)+':act'
        self.write(message)

    def query_channel(self):
        # queries the currently active channel
        # returns the channel as an int
        message = ':serv:chan:act?'
        channel = self.query(message)
        return int(channel)
        
    def set_num_traces(self, num_traces, channel = None):
        # sets the total number of traces present on channel
        # num_traces = int
        # channel = int: if unspecified, defaults to current channel
        if not channel:
            channel = self.query_channel()
        message = ':calc'+str(channel)+':par:coun '+str(num_traces)
        self.write(message)
        
    def query_num_traces(self, channel = None):
        # queries the total number of traces present on channel
        # if channel is unspecified, defaults to the current channel
        if not channel:
            channel = self.query_channel()
        message = ':calc'+str(channel)+':par:coun?'
        num_traces = self.query(message)
        return int(num_traces)   
    
    def allocate_traces(self, alloc_str, channel = None):
        # splits the layout of traces on a given channel
        # if channel is unspecified, defaults to the current channel
        # alloc_str = str
        #   options:
        #       '1'    - one graph in full window
        #       '12'   - two graphs split left/right
        #       '1_2'  - two graphs split top/bottom
        #   see manual for more complicated allocations
        # NOTE: the argument given in the manual has a 'D' in front - this is
        # handled internally by this method
        if not channel:
            channel = self.query_channel()
        message = ':disp:wind'+str(channel)+':spl D'+alloc_str
        self.write(message)
        
    def set_trace(self, trace, channel = None):
        # sets active trace of a given channel
        # if channel is not given, then the current channel is assumed
        # trace = int
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':calc'+str(channel)+':par'+str(trace)+':sel'
        self.write(message)
    
    def query_trace(self, channel = None):
        # queries the active trace on channel
        # channel = int
        # if channel unspecified, defaults to current channel
        # returns trace as an int
        if not channel:
            channel = self.query_channel()
        message = ':serv:chan'+str(channel)+':trac:act?'
        trace = self.query(message)
        return int(trace)
    
    def autoscale(self, channel = None, trace = None):
        # autoscales a given trace on a given channel
        # if channel/trace unspecified, defaults to current
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':disp:wind'+str(channel)+':trac'+str(trace)+':y:auto'
        self.write(message)
    
    ###########
    # MARKERS #
    ###########
    # NOTE: MARKER 10 IS THE REFERENCE MARKER
    
    def toggle_marker(self, marker, state, channel = None):
        # turns marker for a given channel on or off
        # if channel unspecified, defaults to current
        # marker = int
        # state = 0,1 (off,on)
        if not channel:
            channel = self.query_channel()
        message = ':calc'+str(channel)+':mark'+str(marker)+' '+str(state)
        self.write(message)
        
    def move_marker(self, marker, freq, channel = None):
        # move a marker for a given channel to a given frequency
        # if channel unspecified, defaults to current
        # marker = int
        # freq = float (scientific notation OK)
        if not channel:
            channel = self.query_channel()
        message = ':calc'+str(channel)+':mark'+str(marker)+':x '+str(freq)
        self.write(message)
        
    def activate_marker(self, marker, channel = None):
        # activate a marker for a given channel
        # if channel unspecified, defaults to current
        if not channel:
            channel = self.query_channel()
        message = ':calc'+str(channel)+':mark'+str(marker)+':act'
        self.write(message)
        
    def marker_search(self, marker, type_str, channel = None, trace = None):
        # executes marker search type for marker on trace on channel
        # if channel/trace unspecified, defaults to current
        # marker = int
        # type_str = str
        #   options:
        #       'min'   - minimum
        #       'max'   - maximum
        #       'peak'  - peak
        #       'lpe'   - peak to the left
        #       'rpe'   - peak to the right
        #       'targ'  - target
        #       'ltar'  - target to the left
        #       'rtar'  - target to the right
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace)+':mark'+str(marker) \
                        +':func' 
        self.write(message+':type '+type_str)
        self.write(message+':exec')
        
    def marker_track(self, marker, type_str, channel = None, trace = None):
        # performs a marker search every sweep according to the type specified
        # if channel/trace unspecified, defaults to current
        # marker = int
        # type_str = str
        #   options:
        #       'min'   - minimum
        #       'max'   - maximum
        #       'peak'  - peak
        #       'lpe'   - peak to the left
        #       'rpe'   - peak to the right
        #       'targ'  - target
        #       'ltar'  - target to the left
        #       'rtar'  - target to the right
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace)+':mark'+str(marker) \
                        +':func' 
        self.write(message+':type '+type_str)
        self.toggle_marker_tracking(marker, 1, channel, trace)
        
    def toggle_marker_tracking(self, marker, state, channel = None, \
                               trace = None):
        # toggles tracking for a given marker
        # marker = int
        # state = 0,1 (off,on)
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace)+':mark'+str(marker) \
                            +':func:trac '+str(state)
        self.write(message)
        
    def toggle_marker_search_range(self, state, channel = None, trace = None):
        # toggles whether to use a manual search range
        # state = 0,1 (off,on)
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace) \
                        +':mark:func:dom '+str(state)
        self.write(message)
    
    def set_marker_search_start(self, freq, channel = None, trace = None):
        # sets the starting frequency of the marker search domain
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace) \
                        +':mark:func:dom:star '+str(freq)
        self.write(message)
            
    def set_marker_search_stop(self, freq, channel = None, trace = None):
        # sets the starting frequency of the marker search domain
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace) \
                        +':mark:func:dom:stop '+str(freq)
        self.write(message)
        
    def toggle_bandwidth_search(self, state, channel = None, trace = None):
        # toggles bandwidth search mode
        # state = 0,1 (off,on)
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace)+':mark:bwid ' \
                    +str(state)
        self.write(message)
        
    def set_bandwidth_threshold(self, marker, threshold, \
                                    channel = None, trace = None):
        # sets the threshold for determining bandwidth
        # threshold = float : +/- 3dB usually
        # marker = int
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':trac'+str(trace)+':mark'+str(marker) \
                    +':bwid:thr '+str(threshold)
        self.write(message)
    
    def track_resonance(self, marker = 1, channel = None, trace = None):
        # track resonance in the current window, find 3dB points and Q
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        self.toggle_bandwidth_search(1, channel, trace)
        self.set_bandwidth_threshold(marker, +3, channel, trace)
        self.toggle_marker(marker, 1, channel)
        self.activate_marker(marker, channel)
        self.marker_track(marker, 'min', channel, trace)

    ########################
    # MEASUREMENT SETTINGS #
    ########################
    
    def toggle_output(self, state):
        # turns RF Out off or on
        # state = 0,1 (off,on)
        message = ':outp '+str(state)
        self.write(message)
    
    def set_measurement(self, meas_str, channel = None, trace = None):
        # sets the measurement carried out by a trace on a channel
        # meas_str = str
        #   options:
        #       'S11' - reflection on port 1
        #       'S21' - transmission to port 2 from port 1
        #       'S12' - transmission to port 1 from port 2
        #       'S22' - reflection on port 2
        #   see manual for others
        # channel = int
        # trace = int
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':par'+str(trace)+':def '+meas_str
        self.write(message)

    def query_measurement(self, channel = None, trace = None):
        # queries the current measurement carried out by a trace on a channel
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc'+str(channel)+':par'+str(trace)+':def?'
        meas_str = self.query(message)
        return meas_str[:-1]
        
    def set_sweep_type(self, type_str, channel = None):
        # sets the type of sweep on a given channel
        # if channel not specified, defaults to current channel
        # type_str = str
        #   options:
        #       'lin' - linear frequency sweep
        #       'log' - logarithmic frequency sweep
        #       'segm' - segment sweep
        #       'pow' - power sweep
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:type '+type_str
        self.write(message)
        
    def query_sweep_type(self, channel = None):
        # queries type of sweep on channel
        # if channel not specified, defaults to current
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:type?'
        type_str = self.query(message)
        return type_str[:-1]
    
    def set_sweep_segments(self, \
                           freq1_array, \
                           freq2_array, \
                           measurement_points, 
                           interval_type = 'range', \
                           channel = None):
        # interval_type = str: 'range' or 'span'
        #   - range: freq1_array = start vals, freq2_array = stop_vals
        #   - span: freq1_array = center vals, freq2_array = span_vals
        # measurement_points = int or array of ints
        #   if int -> all segments have same # of pts
        #   if array -> different # pts for each segment, len(array)=n_segments
        # OTHER OPTIONS AVAILABLE AND NOT PROGRAMMED
        n_segments = len(freq1_array) # = len(freq2_array)
        if not channel:
            channel = self.query_channel()
        if interval_type == 'range':
            interval_flag = 0
        if interval_type == 'span':
            interval_flag = 1
        if isinstance(measurement_points,type(int(1))):
            n_points = measurement_points
            measurement_points = n_points*np.ones(n_segments,dtype=int)
        message = ':sens'+str(channel)+':segm:data 5,'+str(interval_flag)+\
                    ',0,0,0,0,'+str(n_segments)+','
        for ii in range(n_segments):
            message += str(freq1_array[ii])+','
            message += str(freq2_array[ii])+','
            message += str(measurement_points[ii])
            if ii != n_segments-1:
                message += ','
        self.write(message)
                    
    def toggle_arbitrary_segments(self, state, channel = None):
        # toggles whether segments can be arbitrary
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':segm:arb '+str(state)
        self.write(message)

            
    def set_sweep_mode(self, mode_str, channel = None):
        # sets sweep mode of a channel
        # if channel unspecified, defaults to current
        # channel = int
        # mode_str = str
        #   options (only caps part necessary):
        #       'STEPped'   - stepped mode
        #       'ANALog'    - swept mode
        #   manual specified a couple more options that don't seem to do
        #   anything for this model network analyzer
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:gen '+mode_str
        self.write(message)

    def query_sweep_mode(self, channel = None):
        # queries sweep mode of a channel
        # if channel unspecified, defaults to current
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:gen?'
        mode_str = self.query(message)
        return mode_str[:-1]

    def set_sweep_points(self, points, channel = None):
        # sets the number of points in the sweep on the specified channel
        # if channel unspecified, defaults to current
        # points = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:poin '+str(points)
        self.write(message)
        
    def query_sweep_points(self, channel = None):
        # sets the number of points in the sweep on the specified channel
        # if channel unspecified, defaults to current
        # points = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:poin?'
        return int(self.query(message))

    def toggle_automatic_sweep_time(self, state, channel = None):
        # state = 0,1 for off,on
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:time:auto '+str(state)
        self.write(message)
        
    def set_sweep_time(self, time, channel = None):
        # time = float, in seconds
        if not channel:
            channel = self.query_channel()
        self.toggle_automatic_sweep_time(0,channel=channel)
        message = ':sens'+str(channel)+':swe:time '+str(time)
        self.write(message)
        
    def query_sweep_time(self, channel = None):
        # time = float, in seconds
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':swe:time?'
        return float(self.query(message))

    def set_frequency_start(self, freq, channel = None):
        # sets the start frequency of the sweep on a channel
        # if channel unspecified, defaults to current
        # freq = float: frequency in Hz
        #               scientific notation also OK
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':freq:star '+str(freq)
        self.write(message)
        
    def set_frequency_stop(self, freq, channel = None):
        # sets the stop frequency of the sweep on a channel
        # if channel unspecified, defaults to current
        # freq = float: frequency in Hz
        #               scientific notation also OK
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':freq:stop '+str(freq)
        self.write(message)
        
    def set_frequency_center(self, freq, channel = None):
        # sets the center frequency of the sweep on a channel
        # if channel unspecified, defaults to current
        # freq = float: frequency in Hz
        #               scientific notation also OK
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':freq:cent '+str(freq)
        self.write(message)
        
    def set_frequency_span(self, freq, channel = None):
        # sets the center frequency of the sweep on a channel
        # if channel unspecified, defaults to current
        # freq = float: frequency in Hz
        #               scientific notation also OK
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':freq:span '+str(freq)
        self.write(message)
        
    def set_freqs(self, freq1, freq2, interval_type = 'span', channel = None):
        # sets a frequency interval according to the type
        # interval_type = str: 'range' or 'span'
        #   if range:
        #       freq1 = start
        #       freq2 = stop
        #   if span:
        #       freq1 = center
        #       freq2 = span
        # freq1,2 = float
        if not channel:
            channel = self.query_channel()
        if interval_type == 'range':
            self.set_frequency_start(freq1,channel)
            self.set_frequency_stop(freq2,channel)
        elif interval_type == 'span':
            self.set_frequency_center(freq1,channel)
            self.set_frequency_span(freq2,channel)
        else:
            print('ERROR: Invalid Interval Type!')
                
    def set_power(self, power, channel = None):
        # sets the power output of a given channel
        # if channel unspecified, defaults to current
        # power = float: power in dBm
        if not channel:
            channel = self.query_channel()
        message = ':sour'+str(channel)+':pow '+str(power)
        self.write(message)
                
    def set_format(self, format_str, channel = None, trace = None):
        # sets the measurement format of a given trace on a given channel
        # if channel/trace not specified, defaults to current
        # format_str = str
        #   options:
        #       'MLOGarithmic'  - logarithmic magnitude
        #       'PHASe'         - Phase
        #       'UPHase'        - Expanded phase
        #       'SMITh'         - Smith chart: R+jX (resistance/reactance)
        #       'PPHase'        - Postive phase
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc'+str(channel)+':trac'+str(trace)+':form '+format_str
        self.write(message)
    
    def query_format(self, channel = None, trace = None):
        # queries the format of a given troce on a given channel
        # if channel/trace not specified, defaults to current
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc'+str(channel)+':trac'+str(trace)+':form?'
        format_str = self.query(message)
        return format_str[:-1]
    
    def set_electrical_delay(self, delay, channel = None, trace = None):
        # sets the electrical delay for phase measurements
        # if channel/trace unspecified, defaults to current
        # delay = float: time in seconds (scientific notation OK)
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc'+str(channel)+':trac'+str(trace)+':corr:edel:time ' \
                        + str(delay)
        self.write(message)

    #####################
    # AVERAGING SETINGS #
    #####################
    
    def toggle_averaging(self, state, channel = None):
        # toggle averaging on or off for a given channel
        # if channel unspecified, defaults to current
        # state = int: 0 or 1 (off or on)
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':aver '+str(state)
        self.write(message)
        
    def set_averaging(self, factor, channel = None):
        # set averaging factor for a given channel
        # if channel unspecified, defaults to current
        # factor = int
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':aver:coun '+str(factor)
        self.write(message)
        
    def restart_averaging(self, channel = None):
        # restarts averaging on a given channel
        # if channel not specified, defaults to current
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':aver:cle'
        self.write(message)
        
    def toggle_averaging_trigger(self, state):
        # toggles averaging on trigger initiation
        # usage: collect the avg of many traces after a single trigger
        # state = 0,1 (off,on)
        message = ':trig:aver '+str(state)
        self.write(message)
    
    def set_if_bandwidth(self, bandwidth, channel = None):
        # sets the IF bandwidth of a given channel
        #   -can affect noisiness of measurements - see manual
        # if channel unspecified, defaults to current
        # bandwidth = float (frequency in Hz - scientific notation OK)
        # channel = int
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':band '+str(bandwidth)
        self.write(message)
        
    def toggle_smoothing(self, state, trace = None, channel = None):
        # state = 0,1 (off,on)
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc'+str(channel)+':trac'+str(trace)+':smo '+str(state)
        self.write(message)
        
    def set_smoothing(self, percent, trace = None, channel = None):
        # percent = float, 0.05->25
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc'+str(channel)+':trac'+str(trace)+':smo:aper ' \
                    +str(percent)
        self.write(message)
    
    ####################
    # TRIGGER SETTINGS #
    ####################

    def toggle_continuous_triggering(self, state, channel = None):
        # turns continuous triggering mode on or off for a given channel
        # if channel unspecified, defaults to current
        # state = 0,1 (off,on)
        if not channel:
            channel = self.query_channel()
        message = ':init'+str(channel)+':cont '+str(state)
        self.write(message)
        
    def set_trigger_source(self, source_str):
        # sets the source of triggers
        # source_str = str
        #   options:
        #       'int' - internal (always on, I think, so no real control)
        #       'ext' - external (back panel)
        #       'man' - manual (use this one for taking data)
        #       'bus' - bus (just responds to '*TRG' SCPI command - useless?)
        message = ':trig:sour '+source_str
        self.write(message)

    def set_trigger_scope(self, scope_str):
        # sets whether all channels or just the active channel are triggered
        # scope_str = str: 'all' or 'act'
        message = ':trig:scop '+scope_str
        self.write(message)

    def trigger(self, wait = True):
        # triggers a single measurement
        # can be used in conjunction with a query of '*OPC?' to see when
        #   the measurement has completed
        # wait = 0,1 or False,True
        #   if 1: waits for operation to complete before returning
        message = ':trig:sing'
        self.write(message)
        if wait:
            return self.query('*OPC?')

    ################
    # INPUT/OUTPUT #
    ################

    def transfer_data_to_memory(self, channel = None, trace = None):
        # transfers data trace to memory trace (so it is not overwritten by
        #   subsequent datasets until calling this function again)
        # if channel/trace unspecified, default to current
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc'+str(channel)+':trac'+str(trace)+':math:mem'
        self.write(message)
        
    def get_trace_data(self, \
                       channel = None, \
                       trace = None, \
                       convert = True, \
                       return_array = True, \
                       transfer_first = True,
                       use_memory = True):
        # gets data from the memory trace on the VNA and returns a local copy
        # if channel/trace unspecified, defaults to current
        # if convert is true, will return the trace data in an array of
        #   floats (or complex numbers in the case of a smith chart)
        # NOTE: DOESN'T NECESSARILY SUPPORT ALL TRACE FORMATS CURRENTLY
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
            
        if use_memory:
            if transfer_first:
                self.transfer_data_to_memory(channel=channel, trace=trace)
            message = ':calc'+str(channel)+':trac'+str(trace)+':data:fmem?'
        else:
            message = ':calc'+str(channel)+':trac'+str(trace)+':data:fdat?'
        data_str = self.query(message)

        format_str = self.query_format(channel,trace)
        data_list = data_str.split(',')
        return_list = []
        for ii in range(len(data_list)//2):
            if format_str == 'SMIT':
                resistance = float(data_list[2*ii])
                reactance = float(data_list[((2*ii)+1)])
                if convert:
                    return_list.append(complex(resistance,reactance))
                if not convert:
                    return_list.append(str(complex(resistance,reactance)))
                    return_str = ','.join(return_list)
            else:
                if convert:
                    return_list.append(float(data_list[2*ii]))
                else:
                    return_list.append(data_list[2*ii])
                    return_str = ','.join(return_list)
        if convert:
            if return_array:
                return np.array(return_list)
            else:
                return return_list
        else:
            return return_str

            
    def get_frequency_data(self, channel = None, convert = True, return_array = True):
        # gets all the frequency points for a given channel
        # if convert to floats, will return the trace data
        if not channel:
            channel = self.query_channel()
        message = ':sens'+str(channel)+':freq:data?'
        freq_str = self.query(message)
        if convert:
            freq_list = freq_str.split(',')
            freq_list = [float(ii) for ii in freq_list]
            if return_array:
                return np.array(freq_list)
            else:
                return freq_list
        else:
            return freq_str[:-1]
        
    def get_parameters(self):
        # return all parameters relevant for interpreting trace data
        # power, averaging, electrical delay, format, Sij,
        pass
       

    ##########
    # MACROS #
    ##########
    
    def setup_complex_reflection(self, measurement='S21', autoscale = True):
        # sets up the VNA to measure and display logmag and phase
        self.set_num_traces(2)
        self.allocate_traces('1_2')
        self.set_measurement(measurement,trace=1)
        self.set_measurement(measurement,trace=2)
        self.set_format('mlog',trace=1)
        self.set_format('pphase',trace=2)
        if autoscale:
            self.autoscale(trace=1)
            self.autoscale(trace=2)
            
    def toggle_triggered_operation(self,state):
        # state = 0,1 or False,True
        if state:
            self.set_trigger_source('man')
        else:
            self.set_trigger_source('int')
        self.toggle_averaging_trigger(state)
        
    def generate_cloned_channels(self, channel_alloc_str, n_channels, register_str = 'A'):
        # generate multiple channels identical to the current channel
        self.save_channel_state(register_str = register_str)
        self.allocate_channels(channel_alloc_str)
        # VNA indexes channels from 1
        for ii in range(1,n_channels+1): 
            self.set_channel(ii)
            self.recall_channel_state(register_str = register_str)
        
        

"""
*******************************************************************************
*******************************************************************************
"""      
    
class tektronix_tds7704b(gpib_instrument):
    # OSCILLOSCOPE
    
    def __init__(self, addr):
        super().__init__(addr)

    def toggle_channel(self, channel, state):
        # turns a channel on or off
        # channel = int: 1-4
        # state = 0,1 (off,on)
        message = 'sel:ch' + str(channel) + ' ' + str(state)
        self.write(message)

    def set_coupling(self, channel, coupling_string):
        # sets the coupling for a given channel
        # channel = int: 1-4
        # coupling_string = str: 'dc' or 'gnd'
        message = 'ch' + str(channel) + ':coup ' + coupling_string
        self.write(message)

    #################
    # VERTICAL AXIS #
    #################
    
    def set_vertical_offset(self, channel, offset):
        # sets the offset of a given channel in volts
        # channel = int: 1-4
        # offset = float: in volts, scientific notation OK
        message = 'ch' + str(channel) + ':offs ' + str(offset)
        self.write(message)

    def set_vertical_position(self, channel, divisions):
        # sets the vertical position in divisions
        # channel = int: 1-4
        # divisions = float
        message = 'ch' + str(channel) + ':pos ' + str(divisions)
        self.write(message)

    def set_vertical_scale(self, channel, scale):
        # sets the vertical scale in volts per division
        # channel = int: 1-4
        # scale = float: volts per division, scientific notation OK
        message = 'ch' + str(channel) + ':sca ' + str(scale)
        self.write(message)
        
    ###################
    # HORIZONTAL AXIS #
    ###################
    
    def toggle_horizontal_delay(self, state):
        # turn trigger delay on or off
        # state = 0,1 (off,on)
        message = 'hor:del:mod '+str(state)
        self.write(message)
        
    def set_horizontal_delay(self, delay):
        # sets the trigger delay in seconds
        # delay = float: in seconds, scientific notation OK
        message = 'hor:del:tim '+str(delay)
        self.write(message)
        
    def set_horizontal_position(self, position):
        # sets the horizontal position of the trigger
        # like delay, but when delay is not toggled
        # probably not too useful for actual data collection
        # position = float: percentage of window
        message = 'hor:pos '+str(position)
        self.write(message)
        
    def set_horizontal_reference(self, ref):
        # sets the reference point of the horizontal window
        #   trigger point + delay falls at this point in the horizontal window
        # ref = float: percentage of window, 0 to 100
        message = 'hor:del:pos '+str(ref)
        self.write(message)
        
    def set_horizontal_samplerate(self, rate):
        # NOT INDEPENDENT: AFFECTED BY RECORD LENGTH AND SCALE
        # sets the sample rate in Hz
        # rate = float: in Hz, scientific notation OK
        message = 'hor:mai:sampler '+str(rate)
        self.write(message)
        
    def set_horizontal_scale(self, scale):
        # NOT INDEPENDENT: AFFECTED BY SAMPLE RATE AND RECORD LENGTH
        # sets horizontal scale in seconds per division
        # scale = float: in seconds/division, scientific notation OK
        message = 'hor:sca '+str(scale)
        self.write(message)
        
    def set_horizontal_record_length(self, samples):
        # NOT INDEPENDENT: AFFECTED BY SAMPLE RATE AND SCALE
        # sets number of samples in a record
        # samples = int
        message = 'hor:reco '+str(samples)
        self.write(message)

    def get_horizontal_record_length(self):
        # query horizontal record length
        message = 'hor:reco?'
        return int(self.query(message))
    

    ########################
    # ACQUISITION SETTINGS #
    ########################
    
    def toggle_single_acquisition(self, state):
        # turn single acquisition off or on (off means continuous acquisition)
        # state = 0,1 (off,on)
        if state:
            mode_str = 'seq'
        else:
            mode_str = 'runst'
        message = 'acq:stopa '+mode_str
        self.write(message)
    
    def set_acquisition_sampling_mode(self, mode_str):
        # sets the type of sampling
        # mode_str = str:
        #   options:
        #       'rt'    - real time
        #       'it'    - interpolated time
        #       'et'    - equivalent time
        message = 'acq:samp '+mode_str
        self.write(message)

    ######################
    # FASTFRAME SETTINGS #
    ######################
    
    def toggle_fastframe(self, state):
        # turn fastframe off or on
        # state = int: 0,1 (off,on)
        message = 'hor:fast:state '+str(state)
        self.write(message)
    
    def set_frame_count(self, count):
        # sets the number of frames to acquire
        # count: int
        message = 'hor:fast:coun '+str(count)
        self.write(message)

    def set_frame_record_length(self, samples):
        # sets the number of samples in each frame
        # samples = int
        message = 'hor:fast:len '+str(samples)
        self.write(message)

    ####################
    # TRIGGER SETTINGS #
    ####################

    def trigger(self):
        # force a trigger event
        message = 'trig forc'
        self.write(message)

    def toggle_auto_trigger(self, state):
        # turn auto trigger off or on
        # state = int: 0,1 (off,on)
        if state:
            mode_str = 'auto'
        else:
            mode_str = 'norm'
        message = 'trig:a:mod '+mode_str
        self.write(message)

    def set_trigger_coupling(self, coupling_str):
        # sets the coupling of the main trigger
        # coupling_str = str
        # options: 'ac','dc', see manual for others
        message = 'trig:a:edge:coup '+coupling_str
        self.write(message)

    def set_trigger_slope(self, slope_str):
        # sets whether the trigger event is on a rising or falling edge
        # slope_str = str: 'rise' or 'fall'
        message = 'trig:a:edge:slo '+slope_str
        self.write(message)
        
    def set_trigger_source(self, channel):
        # sets the channel that is the source of the trigger signal
        #   NOTE: might be useful to trigger on AUX IN eventually, but this
        #           isn't currently supported by this method
        # channel = int: 1-4
        message = 'trig:a:edge:sou ch'+str(channel)
        self.write(message)
        
    def set_trigger_holdoff_mode(self, mode_str):
        # sets the type of trigger holdoff
        # mode_str = str: 'time', 'random', or 'auto'
        message = 'trig:a:hold:by '+mode_str
        self.write(message)
        
    def set_trigger_holdoff(self, delay):
        # sets the trigger holdoff in seconds
        # delay = float: scientific notation OK
        message = 'trig:a:hold:tim '+str(delay)
        self.write(message)
        
    def set_trigger_level(self, level=None):
        # sets trigger level in volts
        # level = float: volts, scientific notation OK
        # default behavior (without level passed): set to 50%
        if level:
            message = 'trig:a:lev '+str(level)
        else:
            message = 'trig:a'
        self.write(message)
        
    ################
    # INPUT/OUTPUT #
    ################

    def set_data_encoding(self, encoding_str):
        # sets the format of the output data
        # encoding_str = str
        #   options:
        #       'ascii'
        #       'rib' - a binary data format
        #       see manual for others
        message = 'dat:enc '+encoding_str
        self.write(message)

    def set_data_source(self, channel):
        # sets the channel from which output data will be obtained
        # channel = int: 1-4
        message = 'dat:sou ch'+str(channel)
        self.write(message)

    def get_data(self, channel=None, encoding_str=None):
        # gets data from a given channel
        # if no channel given, assume the current source channel
        # if no encoding given, assume current encoding
        if channel:
            self.set_data_source(channel)
        if encoding_str:
            self.set_data_encoding(encoding_str)
        message = 'curv?'
        return self.query(message)

    def set_data_start(self,sample):
        # sets the first sample to transfer with curv?
        # sample = int: index from 1
        message = 'dat:start '+str(sample)
        self.write(message)
        
    def set_data_stop(self,sample):
        # sets the last sample to transfer with curv?
        # sample = int
        message = 'dat:stop '+str(sample)
        self.write(message)
        
    def get_data_in_chunks(self, chunk_size=None, start=1, stop=None, \
                           channel = None, encoding_str = 'ascii'):
        # get a large data set chunk by chunk and then concatenate
        if channel:
            self.set_data_source(channel)
        self.set_data_encoding(encoding_str)
        if not stop:
            stop = self.get_horizontal_record_length()
        n_samples = stop-start
        if not chunk_size:
            chunk_size = round(n_samples/1000)
            if chunk_size == 0:
                chunk_size = 1
                
        samples_left = n_samples
        data = ''
        first_sample = start
        last_sample = chunk_size
        self.set_data_start(first_sample)
        self.set_data_stop(last_sample)
        while samples_left > 0:
            new_data = self.get_data()
            data += new_data[:-1]+','
            self.wai()
            first_sample += chunk_size
            samples_left -= chunk_size
            if samples_left < chunk_size:
                last_sample = stop
            else:
                last_sample += chunk_size
            self.set_data_start(first_sample)
            self.set_data_stop(last_sample)
            self.wai()
                        
        data = self.convert_ascii_to_volts(data[:-1])
        return data
            
    def get_ymult(self):
        # get factor for scaling ascii to volts
        message = 'wfmo:ymu?'
        return float(self.query(message))
    
    def get_ypos(self):
        # get y position for converting ascii to volts
        message = 'wfmo:yof?'
        return float(self.query(message))
    
    def get_yoff(self):
        # get y offset for converting ascii to volts
        message = 'wfmo:yze?'
        return float(self.query(message))
    
    def convert_ascii_to_volts(self,data_str):
        # convert ascii data to voltages
        data_list = data_str.split(',')
        ymult = self.get_ymult()
        yoff = self.get_yoff()
        ypos = self.get_ypos()
        data_list =[((float(item)-ypos)*ymult)+yoff for item in data_list]
        return np.array(data_list)
    
    def get_tscale(self):
        # get the time scale associated with each sample
        message = 'wfmo:xin?'
        return float(self.query(message))
    
    def get_tstart(self):
        # get the starting time of the data
        message = 'wfmo:xzero?'
        return float(self.query(message))
    
    def get_time_axis(self):
        # get the time axis of an acquisition
        # return as csv string
        tscale = self.get_tscale()
        tstart = self.get_tstart()
        records = self.get_horizontal_record_length()
        total_time = tscale*records
        tstop = tstart + total_time
        time_axis = np.linspace(tstart,tstop,num=records)
        return time_axis
        

"""
*******************************************************************************
*******************************************************************************
"""    

class tektronix_awg7052(gpib_instrument):
    
    def __init__(self,addr):
        super().__init__(addr)

    def trigger(self):
        # force a trigger event
        message = '*trg'
        self.write(message)

    def toggle_output(self,channel,state):
        # toggles whether channel is on or off
        # channel   = int: 1 or 2
        # state = 0,1 (off,on)
        message = 'outp'+str(channel)+' '+str(state)
        self.write(message)

    def run(self):
        # initiate output (turns 'run' on)
        message = 'awgc:run'
        self.write(message)
        
    def stop(self):
        # stops output (turns 'run' off)
        message = 'awgc:stop'
        self.write(message)
        
    def toggle_run(self,state):
        # toggles whether the awg is running or not
        # state = 0,1 (off,on)
        if state:
            self.run()
        else:
            self.stop()

    def set_sampling_rate(self,rate):
        # rate = float, scientific notation x.yEz works too, flexible
        message = 'sour:freq '+str(rate)
        self.write(message)

    def set_run_mode(self,mode):
        # sets the run mode, obviously
        # mode = str
        #   -options:
        #       'cont'  -continuous, repeats waveform indefinitely
        #       'trig'  -triggered, outputs one waveform for each trigger
        #       'gat'   -gated, see manual
        #       'seq'   -sequence, see manual
        #       'enh'   -enhanced, see manual
        message = 'awgc:rmode '+mode
        self.write(message)

    def set_frequency_reference(self,source):
        # sets whether to use the internal or an external 10mhz reference
        # source = str: 'int' or 'ext'
        message = 'sour:rosc:sour '+source
        self.write(message)

    #########################
    # WAVEFORM MANIPULATION #
    #########################

    def new_waveform(self, name, size):
        # creates a new (empty) waveform in the current waveform list
        # name = str
        # size = int, number of samples in waveform
        message = 'wlis:wav:new '+'"'+name+'"'+','+str(size)+',REAL'
        self.write(message)
        
    def delete_waveform(self,name):
        # deletes a waveform from the current list
        # name = str
        message = 'wlis:wav:del '+'"'+name+'"'
        self.write(message)
        
    def clear_waveforms(self):
        # deletes all user-defined waveforms from the current list
        message = 'wlis:wav:del ALL'
        self.write(message)
        
    def send_waveform(self, name, w, m1, m2):
        # name = str
        # w = numpy array of floats
        # m1, m2 = numpy arrays containing only 0s and 1s
        n_samples = len(w)
        self.delete_waveform(name)
        self.new_waveform(name,n_samples)
        
        m = ((2**7)*m2) + ((2**6)*m1)
        
        bytes_data = b''
        for ii in range(n_samples):
            bytes_data += struct.pack('fB',w[ii],int(m[ii]))
        
        num_bytes = n_samples*5
        num_bytes = str(num_bytes)
        num_digits = str(len(num_bytes))
        
        num_bytes = num_bytes.encode('ascii')
        num_digits = num_digits.encode('ascii')
        bytes_count = num_digits + num_bytes
        
        bytes_name = name.encode('ascii')
        bytes_samples = str(n_samples)
        bytes_samples = bytes_samples.encode('ascii')
        
        message = b'wlis:wav:data "'+bytes_name+b'",0,'+bytes_samples \
                    +b',#'+bytes_count+bytes_data
        self.write_raw(message)

    def load_waveform(self,channel,name):
        # load a waveform from the current list into channel
        # channel = int: 1 or 2
        # name = str
        message = 'sour'+str(channel)+':wav "'+name+'"'
        self.write(message)

    #######################
    # ANALOG OUT SETTINGS #
    #######################
        
    def set_analog_amplitude(self, channel, amplitude, units='V'):
        # sets the peak to peak voltage amplitude of a given channel
        # channel   = int: 1 or 2
        # amplitude = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':volt '+str(amplitude)+units
        self.write(message)

    ###################
    # MARKER SETTINGS #
    ###################

    def set_marker_low(self, channel, marker, voltage, units='V'):
        # set the low voltage of a given marker on a given channel
        # channel   = int: 1 or 2
        # marker    = int: 1 or 2
        # voltage = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':mark'+str(marker)+':volt:low ' \
                    + str(voltage)+units
        self.write(message)

    def set_marker_high(self, channel, marker, voltage, units='V'):
        # set the low voltage of a given marker on a given channel
        # channel   = int: 1 or 2
        # marker    = int: 1 or 2
        # voltage = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':mark'+str(marker)+':volt:high ' \
                    + str(voltage)+units
        self.write(message)
        
    def set_markers_low(self, channel, voltage, units='V'):
        # same as above, but both markers for a given channel
        self.set_marker_low(channel,1,voltage,units)
        self.set_marker_low(channel,2,voltage,units)        

    def set_markers_high(self, channel, voltage, units='V'):
        # same as above, but both markers for a given channel
        self.set_marker_high(channel,1,voltage,units)
        self.set_marker_high(channel,2,voltage,units)

    def set_marker_delay(self, channel, marker, delay):
        # set the delay for a given channel and marker
        # NOTE: ONLY ACCEPTS DELAY IN PICOSECONDS
        # channel   = int: 1 or 2
        # marker    = int: 1 or 2
        # delay     = float: time in picoseconds
        # units     = str
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':del ' \
                    + str(delay) + 'ps'
        self.write(message)

    #######################
    # FILE SYSTEM METHODS #
    #######################

    def query_cwd(self):
        # return current working directory of awg mass memory
        message = 'mmem:cdir?'
        return self.query(message)

    def mkdir(self, dir_name):
        # makes a new directory in the current working directory
        # dir_name = str
        message = 'mmem:mdir '+'"'+dir_name+'"'
        self.write(message)
        
    def ls(self):
        # query contents of current working directory
        return self.query('mmem:cat?')

    def cd(self,rel_path):
        # change directory relative to current directory
        # rel_path = str
        # '..' moves up one level
        message = 'mmem:cdir '+'"'+rel_path+'"'
        self.write(message)

    def reset_cwd(self):
        message = 'mmem:cdir'
        self.write(message)

    def set_cwd(self,absolute_path):
        # set current working directory of awg mass memory to absolute_path
        # path = str
        self.reset_cwd()
        message = 'mmem:cdir '+'"'+absolute_path+'"'
        self.write(message)

"""
*******************************************************************************
*******************************************************************************
"""

class tektronix_awg520(gpib_instrument):
    
    def __init__(self,addr):
        super().__init__(addr)

    def trigger(self):
        # force a trigger event
        message = '*trg'
        self.write(message)

    def toggle_output(self,channel,state):
        # toggles whether channel is on or off
        # channel   = int: 1 or 2
        # state = 0,1 (off,on)
        message = 'outp'+str(channel)+' '+str(state)
        self.write(message)
    
    def set_run_mode(self,mode):
        # sets the run mode, obviously
        # mode = str
        #   -options:
        #       'cont'  -continuous, repeats waveform indefinitely
        #       'trig'  -triggered, outputs one waveform for each trigger
        #       'gat'   -gated, see manual
        #       'enh'   -enhanced, see manual
        message = 'awgc:rmode '+mode
        self.write(message)
        
    def run(self):
        # initiate output (turns 'run' on)
        message = 'awgc:run'
        self.write(message)
        
    def stop(self):
        # stops output (turns 'run' off)
        message = 'awgc:stop'
        self.write(message)
        
    def toggle_run(self,state):
        # toggles whether the awg is running or not
        # state = 0,1 (off,on)
        if state:
            self.run()
        else:
            self.stop()
            
    def set_offset(self, channel, offset, units='V'):
        # set the voltage offset of a given channel
        # channel   = int: 1 or 2
        # offset = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':volt:offs '+str(offset)+units
        self.write(message)
        
    def set_amplitude(self, channel, amplitude, units='V'):
        # sets the peak to peak voltage amplitude of a given channel
        # channel   = int: 1 or 2
        # amplitude = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':volt '+str(amplitude)+units
        self.write(message)

    def set_frequency_reference(self, channel, source):
        # sets the source of the 10MHz reference to use with a channel
        # channel = int: 1 or 2
        # source = str: 'int' or 'ext'
        message = 'sour'+str(channel)+':rosc:sour '+source
        self.write(message)

    def send_waveform(self, w, m1, m2, filename, samplerate):
        # w = array of floats between -1 and 1
        # m1 = array of integers, only 0 and 1 allowed
        # m2 as m1
        # filename = str ending in .wfm
        # samplerate = float up to 1.0E9 (one GS/s)
        
        # need to send a block of bytes data with format
        # cmd + file_counter + header + data_counter + bytes_data + trailer
        
        #converting strings to raw bytes data
        bytes_filename = filename.encode('ascii')
        cmd = b'mmem:data "'+bytes_filename+b'",'
        header = b'MAGIC 1000\r\n'
        
        nsamples = len(w)
        m = m1 + np.multiply(m2, 2)
        
        #packing waveform and markers into bytes data
        bytes_data = b''
        for ii in range(nsamples):
            bytes_data += struct.pack('<fB',w[ii],int(m[ii]))
            
        num_bytes = len(bytes_data)
        num_digits = len(str(num_bytes))
        
        # converting this info into bytes
        num_digits = str(num_digits)
        num_digits = num_digits.encode('ascii')
        num_bytes = str(num_bytes)
        num_bytes = num_bytes.encode('ascii')
        data_counter = b'#'+num_digits+num_bytes
        
        samplerate_str = '{:.2E}'.format(samplerate)
        samplerate_bytes = samplerate_str.encode('ascii')
        trailer = b'CLOCK '+samplerate_bytes+b'\r\n'
        
        file = header + data_counter + bytes_data + trailer
        
        num_file_bytes = len(file)
        num_file_digits = len(str(num_file_bytes))
        
        num_file_digits = str(num_file_digits)
        num_file_digits = num_file_digits.encode('ascii')
        num_file_bytes = str(num_file_bytes)
        num_file_bytes = num_file_bytes.encode('ascii')
        file_counter = b'#'+num_file_digits+num_file_bytes
        
        message = cmd + file_counter + file
        self.write_raw(message)

    def load_waveform(self, channel, filename):
        # loads the waveform at filename into a channel
        # channel   = int: 1 or 2
        # filename  = str
        #   - can be a path, always relative to current working directory
        message = 'sour'+str(channel)+':func:user '+'"'+filename+'"'
        self.write(message)

    ###################
    # MARKER SETTINGS #
    ###################

    def set_marker_low(self, channel, marker, voltage, units='V'):
        # set the low voltage of a given marker on a given channel
        # channel   = int: 1 or 2
        # marker    = int: 1 or 2
        # voltage = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':mark'+str(marker)+':volt:low ' \
                    + str(voltage)+units
        self.write(message)

    def set_marker_high(self, channel, marker, voltage, units='V'):
        # set the low voltage of a given marker on a given channel
        # channel   = int: 1 or 2
        # marker    = int: 1 or 2
        # voltage = float
        # units     = str: 'V', 'mV'
        message = 'sour'+str(channel)+':mark'+str(marker)+':volt:high ' \
                    + str(voltage)+units
        self.write(message)

    def set_markers_low(self, channel, voltage, units='V'):
        # same as above, but both markers for a given channel
        self.set_marker_low(channel,1,voltage,units)
        self.set_marker_low(channel,2,voltage,units)        

    def set_markers_high(self, channel, voltage, units='V'):
        # same as above, but both markers for a given channel
        self.set_marker_high(channel,1,voltage,units)
        self.set_marker_high(channel,2,voltage,units)

    def set_marker_delay(self, channel, marker, delay, units='s'):
        # set the delay for a given channel and marker
        # channel   = int: 1 or 2
        # marker    = int: 1 or 2
        # delay     = float
        # units     = str
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':del ' \
                    + str(delay) + units
        self.write(message)

    #######################
    # FILE SYSTEM METHODS #
    #######################

    def set_mass_storage(self, device = 'MAIN'):
        #sets mass storage device
        #device = str
        #options: 'MAIN', 'FLOP', 'NET1', 'NET2', 'NET3'
        message = 'mmem:msis '+'"'+device+'"'
        self.write(message)

    def query_cwd(self):
        # return current working directory of awg mass memory
        message = 'mmem:cdir?'
        return self.query(message)

    def mkdir(self, dir_name):
        # makes a new directory in the current working directory
        # dir_name = str
        message = 'mmem:mdir '+'"'+dir_name+'"'
        self.write(message)
        
    def ls(self):
        # query contents of current working directory
        return self.query('mmem:cat?')

    def cd(self,rel_path):
        # change directory relative to current directory
        # rel_path = str
        # '..' moves up one level
        message = 'mmem:cdir '+'"'+rel_path+'"'
        self.write(message)

    def reset_cwd(self):
        message = 'mmem:cdir'
        self.write(message)

    def set_cwd(self,absolute_path):
        # set current working directory of awg mass memory to absolute_path
        # path = str
        self.reset_cwd()
        message = 'mmem:cdir '+'"'+absolute_path+'"'
        self.write(message)
   

"""
*******************************************************************************
*******************************************************************************
"""

class stanford_sr830(gpib_instrument):
    #LOCK-IN AMPLIFIER
    def __init__(self,addr):
        super().__init__(addr)
        
    def set_interface(self,interface_str='gpib'):
        #interface_str = 'gpib' or 'rs232'
        if interface_str == 'gpib':
            message = 'outx 1'
        elif interface_str == 'rs232':
            message = 'outx 0'
        self.write(message)
        
    def auto_gain(self):
        message = 'agan'
        self.write(message)
        
    def auto_reserve(self):
        message = 'arsv'
        self.write(message)
        
    def auto_phase(self):
        message = 'aphs'
        self.write(message)
        
    def auto_offset(self,quantity_str):
        # quantity_str = 'x','y',or 'r'
        message = 'aoff '
        if quantity_str == 'x':
            message += '1'
        elif quantity_str == 'y':
            message += '2'
        elif quantity_str == 'r':
            message += '3'
        self.write(message)
        
    def set_time_constant(self, time_constant):
        # time_constant = float, discrete set of allowed values
        message = 'oflt '
        list_of_times = [10.0E-6, 30.0E-6, 100.0E-6, 300.0E-6,
                         1.0E-3, 3.0E-3, 10.0E-3, 30.0E-3, 100.0E-3, 300.0E-3,
                         1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                         1.0E3, 3.0E3, 10.0E3, 30.0E3]
        if time_constant not in list_of_times:
            print('INVALID TIME CONSTANT!')
        else:
            message += str(list_of_times.index(time_constant))
        self.write(message)
        
    def get_time_constant(self):
        # time_constant = float, discrete set of allowed values
        message = 'oflt?'
        ind = int(self.query(message))
        list_of_times = [10.0E-6, 30.0E-6, 100.0E-6, 300.0E-6,
                         1.0E-3, 3.0E-3, 10.0E-3, 30.0E-3, 100.0E-3, 300.0E-3,
                         1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                         1.0E3, 3.0E3, 10.0E3, 30.0E3]
        time_constant = list_of_times[ind]
        return time_constant
        
    def set_sample_rate(self, rate):
        # rate = float
        message = 'srat '
        list_of_rates = [62.5E-3, 125.0E-3, 250.0E-3, 500.0E-3,
                         1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                         128.0, 256.0, 512.0]
        if rate not in list_of_rates:
            print('INVALID SAMPLE RATE!')
        else:
            message += str(list_of_rates.index(rate))
        self.write(message)
        
    def get_sample_rate(self):
        message = 'srat?'
        ind = int(self.query(message))
        list_of_rates = [62.5E-3, 125.0E-3, 250.0E-3, 500.0E-3,
                         1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                         128.0, 256.0, 512.0]
        rate = list_of_rates[ind]
        return rate
    
    def set_acquisition_mode(self, mode_str = 'single'):
        # mode_str = 'single' or 'loop'
        message = 'send '
        if mode_str == 'single':
            message += '0'
        elif mode_str == 'loop':
            message += '1'
        self.write(message)
        
    def toggle_triggered_acquisition(self,state):
        # state = 0,1
        message = 'tstr '+str(state)
        self.write(message)
        
    def trigger(self):
        message = 'trig'
        self.write(message)
        
    def set_display_mode(self, mode_str = 'polar'):
        # mode_str = 'polar' or 'cartesian' for r,theta or x,y
        message = 'ddef '
        if mode_str == 'polar':
            message1 = message + '1,1,0'
            message2 = message+ '2,1,0'
        elif mode_str == 'cartesian':
            message1 = message + '1,0,0'
            message2 = message + '2,0,0'
        self.write(message1)
        self.write(message2)
        
    def get_num_buffer_points(self):
        # gets the number of points stored in buffer
        message = 'spts?'
        num = int(self.query(message))
        return num
    
    def reset_buffers(self):
        # resets buffers
        message = 'rest'
        self.write(message)
        
    def get_buffer_data(self, channel, n_points=16383):
        # gets channel data from buffer
        # channel = 1,2 int
        # n_points <= 16383
        """
        message = 'trca ? '+str(channel)+',0,'+str(n_points)
        data = (self.query(message)[:-2]).split(',')
        data = np.array([np.float(item) for item in data])
        """
        message = 'trcl ? '+str(channel)+',0,'+str(n_points)
        self.write(message)
        bytes_data = self.read_raw()
        data = []
        for ii in range(n_points):
            mantissa,exp = struct.unpack('hh',bytes_data[4*ii:(4*ii)+4])
            data.append(mantissa*(2**(exp-124)))
        return np.array(data)
    
    def acquire_data(self, n_points=16383):
        self.reset_buffers()
        self.trigger()
        rate = self.get_sample_rate()
        acquisition_time = n_points/rate
        time.sleep(acquisition_time)
        while True:
            num_points = self.get_num_buffer_points()
            if num_points >= n_points:
                break
        data1 = self.get_buffer_data(1,n_points)
        data2 = self.get_buffer_data(2,n_points)
        time_axis = np.linspace(0,(n_points-1.0)/rate,n_points)
        return data1, data2, time_axis
    
    def configure_acquisition(self, 
                              display_mode = 'polar',
                              sample_rate = 512.0,
                              timeout = 30000):
        self.set_interface('gpib')
        self.set_acquisition_mode('single')
        self.toggle_triggered_acquisition(1)
        self.set_display_mode(display_mode)
        self.set_sample_rate(sample_rate)
        self.set_timeout(timeout)
        
        
    
     
"""
*******************************************************************************
*******************************************************************************
"""

class national_instruments_bnc2090:
    #DAQ
    def __init__(self):
        self.output_voltages = []
        for ii in range(2):
            task = nidaqmx.Task()
            ch_str = 'Dev1/ao' + str(ii)
            task.ao_channels.add_ao_voltage_chan(ch_str)
            self.output_voltages.append(0.0)
            task.write(0.0)
            task.close()

    def set_voltage(self, output_ind, voltage):
        task = nidaqmx.Task()
        ch_str = 'Dev1/ao' + str(output_ind)
        task.ao_channels.add_ao_voltage_chan(ch_str)
        self.output_voltages[output_ind] = voltage
        task.write(voltage)
        task.close()

    def get_voltage(self, input_ind, samples=None):
        task = nidaqmx.Task()
        ch_str = 'Dev1/ai' + str(input_ind)
        task.ai_channels.add_ai_voltage_chan(ch_str)
        if samples:
            result = task.read(samples)
        else:
            result = task.read()
        task.close()
        return result

    def get_mean_voltage(self, input_ind, samples, return_stdev=False):
        voltages = self.get_voltage(input_ind, samples)
        mean = statistics.mean(voltages)
        if return_stdev:
            stdev = statistics.stdev(voltages)
            return (
             mean, stdev)
        else:
            return mean

    def gui(self):
        window = tk.Tk()
        window.geometry('710x450')
        window.title('Analog Outputs')
        window.focus_force()
        
        frame0 = tk.Frame(window,width=800,height=300,bd=3,relief='ridge')
        frame0.grid(row=0,column=0,padx=10,pady=10)
        
        ao0 = tk.DoubleVar()
        ao0.set(0.0)
        
        ao0_coarse = tk.DoubleVar()
        ao0_coarse.set(0.0)
        
        ao0_fine = tk.DoubleVar()
        ao0_fine.set(0.0)
        
        def update_voltage_ao0(*args):
            coarse_voltage = ao0_coarse.get()
            fine_voltage = ao0_fine.get()*0.001
            voltage = round(coarse_voltage+fine_voltage,3)
            ao0.set(voltage)
            self.set_voltage(0,voltage)
        
        def coarse_increment_ao0():
            coarse_val = ao0_coarse.get()+0.1
            ao0_coarse.set(coarse_val)
            update_voltage_ao0()
            
        def coarse_decrement_ao0():
            coarse_val = ao0_coarse.get()-0.1
            ao0_coarse.set(coarse_val)
            update_voltage_ao0()
            
        def fine_increment_ao0():
            fine_val = ao0_fine.get()+1
            ao0_fine.set(fine_val)
            update_voltage_ao0()
            
        def fine_decrement_ao0():
            fine_val = ao0_fine.get()-1
            ao0_fine.set(fine_val)
            update_voltage_ao0()
        
        
        ao0_lbl = tk.Label(frame0, text = 'Analog Out 0', font='helvetica 16 bold')
        ao0_lbl.grid(row=0,column=0, columnspan=2, sticky= 's', padx=10,pady=10)
        
        ao0_box = tk.Entry(frame0, textvariable=ao0,\
                               width=10)
        ao0_box.grid(row=1,column=0, sticky = 'n', pady=10)
        
        box_label = tk.Label(frame0, text = 'Volts')
        box_label.grid(row=1,column=1,sticky='wn',pady=10)
        
        out_coarse_scale = tk.Scale(frame0, from_=-10, to=10, orient='horizontal', \
                             resolution = 0.1, command = update_voltage_ao0, \
                             digits = 3, label = 'V', length = 400, \
                             tickinterval = 2.5, variable = ao0_coarse)
        out_coarse_scale.grid(row=0,column=2,padx=10,pady=10)
        
        out_fine_scale = tk.Scale(frame0, from_=-100, to=100, orient='horizontal', \
                             resolution = 1, command = update_voltage_ao0, \
                             digits = 3, label = 'mV', length = 400, \
                             tickinterval = 25, variable = ao0_fine)
        out_fine_scale.grid(row=1,column=2,padx=10,pady=10)
        
        
        
        coarse_dec_button = tk.Button(frame0,text='-',command=coarse_decrement_ao0,\
                                      font = 'helvetica 16')
        coarse_dec_button.grid(row=0,column=3,sticky='ew',padx=10,pady=10)
        coarse_inc_button = tk.Button(frame0,text='+',command=coarse_increment_ao0,\
                                      font = 'helvetica 16')
        coarse_inc_button.grid(row=0,column=4,sticky='ew',padx=10,pady=10)
        
        
        fine_dec_button = tk.Button(frame0,text='-',command=fine_decrement_ao0,\
                                      font = 'helvetica 16')
        fine_dec_button.grid(row=1,column=3,sticky='ew',padx=10,pady=10)
        fine_inc_button = tk.Button(frame0,text='+',command=fine_increment_ao0,\
                                      font = 'helvetica 16')
        fine_inc_button.grid(row=1,column=4,sticky='ew',padx=10,pady=10)
        
        ao0.trace('w',update_voltage_ao0)
        
        """
        ************************************************
        """
        
        frame1 = tk.Frame(window,width=800,height=300,bd=3,relief='ridge')
        frame1.grid(row=1,column=0,padx=10,pady=10)
        
        ao1 = tk.DoubleVar()
        ao1.set(0.0)
        
        ao1_coarse = tk.DoubleVar()
        ao1_coarse.set(0.0)
        
        ao1_fine = tk.DoubleVar()
        ao1_fine.set(0.0)
        
        def update_voltage_ao1(*args):
            coarse_voltage = ao1_coarse.get()
            fine_voltage = ao1_fine.get()*0.001
            voltage = round(coarse_voltage+fine_voltage,3)
            ao1.set(voltage)
            self.set_voltage(1,voltage)
        
        def coarse_increment_ao1():
            coarse_val = ao1_coarse.get()+0.1
            ao1_coarse.set(coarse_val)
            update_voltage_ao1()
            
        def coarse_decrement_ao1():
            coarse_val = ao1_coarse.get()-0.1
            ao1_coarse.set(coarse_val)
            update_voltage_ao1()
            
        def fine_increment_ao1():
            fine_val = ao1_fine.get()+1
            ao1_fine.set(fine_val)
            update_voltage_ao1()
            
        def fine_decrement_ao1():
            fine_val = ao1_fine.get()-1
            ao1_fine.set(fine_val)
            update_voltage_ao1()
        
        
        ao1_lbl = tk.Label(frame1, text = 'Analog Out 1', font='helvetica 16 bold')
        ao1_lbl.grid(row=3,column=0, columnspan=2, sticky= 's', padx=10,pady=10)
        
        ao1_box = tk.Entry(frame1, textvariable=ao1,\
                               width=10)
        ao1_box.grid(row=4,column=0, sticky = 'n', pady=10)
        
        box_label = tk.Label(frame1, text = 'Volts')
        box_label.grid(row=4,column=1,sticky='wn',pady=10)
        
        out_coarse_scale = tk.Scale(frame1, from_=-10, to=10, orient='horizontal', \
                             resolution = 0.1, command = update_voltage_ao1, \
                             digits = 3, label = 'V', length = 400, \
                             tickinterval = 2.5, variable = ao1_coarse)
        out_coarse_scale.grid(row=3,column=2,padx=10,pady=10)
        
        out_fine_scale = tk.Scale(frame1, from_=-100, to=100, orient='horizontal', \
                             resolution = 1, command = update_voltage_ao1, \
                             digits = 3, label = 'mV', length = 400, \
                             tickinterval = 25, variable = ao1_fine)
        out_fine_scale.grid(row=4,column=2,padx=10,pady=10)
        
        
        
        coarse_dec_button = tk.Button(frame1,text='-',command=coarse_decrement_ao1,\
                                      font = 'helvetica 16')
        coarse_dec_button.grid(row=3,column=3,sticky='ew',padx=10,pady=10)
        coarse_inc_button = tk.Button(frame1,text='+',command=coarse_increment_ao1,\
                                      font = 'helvetica 16')
        coarse_inc_button.grid(row=3,column=4,sticky='ew',padx=10,pady=10)
        
        
        fine_dec_button = tk.Button(frame1,text='-',command=fine_decrement_ao1,\
                                      font = 'helvetica 16')
        fine_dec_button.grid(row=4,column=3,sticky='ew',padx=10,pady=10)
        fine_inc_button = tk.Button(frame1,text='+',command=fine_increment_ao1,\
                                      font = 'helvetica 16')
        fine_inc_button.grid(row=4,column=4,sticky='ew',padx=10,pady=10)
        
        ao1.trace('w',update_voltage_ao1)
        
        window.mainloop()
        
        
"""
*******************************************************************************
*******************************************************************************
"""      
        
class alazartech_ats9462:
    # DIGITIZER
    def __init__(self, \
                 input_range = 4.0, \
                 BW_limit = 0, \
                 clock_source = 'ext', \
                 trigger_delay = 0, \
                 sample_rate = 180.0E6):
        # input_range = float, |voltage| limit in volts
        #               can pass list of floats to set channels A,B separately
        # clock_source = str: 'int' or 'ext'
        # BW_limit = 0, 1, or [chA_flag,chB_flag]
        # trigger delay in seconds
        
        self.board = ats.Board(systemId = 1, boardId = 1)
        
        # internal: ats.INTERNAL_CLOCK
        #   - sample rate -> flag from atsapi
        # external ref: ats.EXTERNAL_CLOCK_10MHz_REF
        #   - sample rate -> int in Hz
        self.sample_rate = sample_rate # NEED TO CHANGE BOTH THIS AND THE ATS FLAG
        if clock_source == 'ext':
            clock_source_code = ats.EXTERNAL_CLOCK_10MHz_REF
            sample_rate_code = int(self.sample_rate)
        elif clock_source == 'int':
            clock_source_code = ats.INTERNAL_CLOCK
            list_of_rates = [180.0E6, 160.0E6, 125.0E6, 100.0E6, 50.0E6, \
                             20.0E6, 10.0E6, 5.0E6, 2.0E6, 1.0E6, 500.0E3, \
                             200.0E3, 100.0E3, 50.0E3, 20.0E3, 10.0E3, \
                             5.0E3, 2.0E3, 1.0E3]
            if sample_rate not in list_of_rates:
                print('INVALID SAMPLE RATE!')
            else:
                list_of_rate_codes = [ats.SAMPLE_RATE_180MSPS, ats.SAMPLE_RATE_160MSPS,\
                                      ats.SAMPLE_RATE_125MSPS, ats.SAMPLE_RATE_100MSPS,\
                                      ats.SAMPLE_RATE_50MSPS, ats.SAMPLE_RATE_20MSPS,\
                                      ats.SAMPLE_RATE_10MSPS, ats.SAMPLE_RATE_5MSPS,\
                                      ats.SAMPLE_RATE_2MSPS, ats.SAMPLE_RATE_1MSPS,\
                                      ats.SAMPLE_RATE_500KSPS, ats.SAMPLE_RATE_200KSPS,\
                                      ats.SAMPLE_RATE_100KSPS, ats.SAMPLE_RATE_50KSPS,\
                                      ats.SAMPLE_RATE_20KSPS, ats.SAMPLE_RATE_10KSPS,\
                                      ats.SAMPLE_RATE_5KSPS, ats.SAMPLE_RATE_2KSPS,\
                                      ats.SAMPLE_RATE_1KSPS]
                sample_rate_code = list_of_rate_codes[list_of_rates.index(sample_rate)]
        else:
            print('INVALID CLOCK SOURCE!')
        clock_edge = ats.CLOCK_EDGE_RISING
        decimation = 0
        self.clock_params = (clock_source_code, sample_rate_code, clock_edge, decimation)


        list_of_ranges = [0.2,0.4,0.8,2.0,4.0]
        if isinstance(input_range,type([])):
            if input_range[0] in list_of_ranges:
                self.channel_A_input_range = input_range[0]
            else:
                print('INVALID CHANNEL A INPUT RANGE!')
            if input_range[1] in list_of_ranges:
                self.channel_B_input_range = input_range[1]
            else:
                print('INVALID CHANNEL B INPUT RANGE!')
        elif isinstance(input_range,type(float(0.5))):
            if input_range in list_of_ranges:
                self.channel_A_input_range = input_range
                self.channel_B_input_range = input_range
            else:
                print('INVALID INPUT RANGE!')
        else:
            print('INVALID DATA TYPE PASSED FOR INPUT RANGE PARAMETER!')
            
        if self.channel_A_input_range == 0.2:
            channel_A_input_range_code = ats.INPUT_RANGE_PM_200_MV
        elif self.channel_A_input_range == 0.4:
            channel_A_input_range_code = ats.INPUT_RANGE_PM_400_MV
        elif self.channel_A_input_range == 0.8:
            channel_A_input_range_code = ats.INPUT_RANGE_PM_800_MV
        elif self.channel_A_input_range == 2.0:
            channel_A_input_range_code = ats.INPUT_RANGE_PM_2_V
        elif self.channel_A_input_range == 4.0:
            channel_A_input_range_code = ats.INPUT_RANGE_PM_4_V

        if self.channel_B_input_range == 0.2:
            channel_B_input_range_code = ats.INPUT_RANGE_PM_200_MV
        elif self.channel_B_input_range == 0.4:
            channel_B_input_range_code = ats.INPUT_RANGE_PM_400_MV
        elif self.channel_B_input_range == 0.8:
            channel_B_input_range_code = ats.INPUT_RANGE_PM_800_MV
        elif self.channel_B_input_range == 2.0:
            channel_B_input_range_code = ats.INPUT_RANGE_PM_2_V
        elif self.channel_B_input_range == 4.0:
            channel_B_input_range_code = ats.INPUT_RANGE_PM_4_V
            
        
        channel_A_id = ats.CHANNEL_A
        channel_A_coupling = ats.AC_COUPLING
        channel_A_impedence = ats.IMPEDANCE_50_OHM
        self.channel_A_params = (channel_A_id, channel_A_coupling, \
                            channel_A_input_range_code, channel_A_impedence)
    
        channel_B_id = ats.CHANNEL_B
        channel_B_coupling = ats.DC_COUPLING
        channel_B_impedence = ats.IMPEDANCE_50_OHM
        self.channel_B_params = (channel_B_id, channel_B_coupling, \
                            channel_B_input_range_code, channel_B_impedence)
        
        if isinstance(BW_limit,type([])):
            self.channel_A_bandwidth = BW_limit[0]
            self.channel_B_bandwidth = BW_limit[1]
        elif isinstance(BW_limit,type(int(2))):
            self.channel_A_bandwidth = BW_limit
            self.channel_B_bandwidth = BW_limit
        else:
            print('INVALID BW LIMIT FLAG!')
            self.channel_A_bandwidth = 0
            self.channel_B_bandwidth = 0
    
        trigger_engine_operation = ats.TRIG_ENGINE_OP_J
        first_engine = ats.TRIG_ENGINE_J
        first_source = ats.TRIG_EXTERNAL
        first_slope = ats.TRIGGER_SLOPE_POSITIVE
        first_level = 190
        second_engine = ats.TRIG_ENGINE_K
        second_source = ats.TRIG_DISABLE
        second_slope = ats.TRIGGER_SLOPE_POSITIVE
        second_level = 128
        self.trigger_params = (trigger_engine_operation, \
                          first_engine, first_source, first_slope, first_level, \
                          second_engine, second_source, second_slope, second_level)
    
        external_trigger_coupling = ats.DC_COUPLING
        external_trigger_input_range = ats.ETR_1V
        self.external_trigger_params = (external_trigger_coupling, \
                                   external_trigger_input_range)
    
        self.trigger_delay_samples = math.ceil(trigger_delay*self.sample_rate)
    
        triggerTimeout_sec = 0
        self.trigger_timeout = int(triggerTimeout_sec / 10e-6 + 0.5)
        
        aux_mode = ats.AUX_OUT_PACER
        aux_param = 18
        self.aux_io_params = (aux_mode, aux_param)
        
        self.configure()
        
    def configure(self):
        self.board.setCaptureClock(*self.clock_params)
        self.board.inputControlEx(*self.channel_A_params)
        self.board.setBWLimit(ats.CHANNEL_A, self.channel_A_bandwidth)
        self.board.inputControlEx(*self.channel_B_params)
        self.board.setBWLimit(ats.CHANNEL_B, self.channel_B_bandwidth)
        self.board.setTriggerOperation(*self.trigger_params)
        self.board.setExternalTrigger(*self.external_trigger_params)
        self.board.setTriggerDelay(self.trigger_delay_samples)
        self.board.setTriggerTimeOut(self.trigger_timeout)
        self.board.configureAuxIO(*self.aux_io_params)
                
    def acquire_NPT_single(self, 
                record_length = 10.0E-6, # in seconds
                triggering_instrument = None, # instrument that sends trigger to start acquisition
                channel_str = 'A', # 'A', 'B', or 'AB':
                acquisition_timeout = 15.0E3, # timeout in ms
                excess_samples = 45, #extra samples to throw out
                benchmark = False):
        #EXCESS SAMPLES REASON:
        #   often there is a saturation of the digitizer right at the end of
        #   an acquisition, so by default I acquire excess samples and throw
        #   them out at the end.
        #   * This will obviously not be suitable for all applications
        
        # NPT = no pre-trigger samples
        preTriggerSamples = 0
        postTriggerSamples = math.ceil(record_length*self.sample_rate) + excess_samples
        
        # Single acquisition: 1 record, 1 buffer
        recordsPerBuffer = 1
        buffersPerAcquisition = 1
        n_buffers = 1
                
        # TODO: Select the active channels.
        if channel_str == 'A':
            channels = ats.CHANNEL_A
        elif channel_str == 'B':
            channels = ats.CHANNEL_B
        elif channel_str == 'AB':
            channels = ats.CHANNEL_A | ats.CHANNEL_B
        else:
            sys.exit('ERROR: Invalid channel string')

        channelCount = 0
        for c in ats.channels:
            channelCount += (c & channels == c)
        
        # Compute the number of bytes per record and per buffer
        memorySize_samples, bitsPerSample = self.board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        samplesPerRecord = preTriggerSamples + postTriggerSamples
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount
            
        sample_type = ctypes.c_uint8
        if bytesPerSample > 1:
            sample_type = ctypes.c_uint16
            
        buffers = []
        for i in range(n_buffers):
            buffers.append(ats.DMABuffer(self.board.handle, sample_type, \
                                         bytesPerBuffer))
        
        # Set the record size
        self.board.setRecordSize(preTriggerSamples, postTriggerSamples)
    
        recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition
        
        # Configure the board to make an NPT AutoDMA acquisition
        self.board.beforeAsyncRead(channels,
                              -preTriggerSamples,
                              samplesPerRecord,
                              recordsPerBuffer,
                              recordsPerAcquisition,
                              ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT)
        
        # Post DMA buffers to board
        for buffer in buffers:
            self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        
        start = time.clock() # Keep track of when acquisition starteda
        try:
            self.board.startCapture() # Start the acquisition
            t0 = time.time()
            print("Capturing %d buffers." %
                  buffersPerAcquisition)
            buffersCompleted = 0
            bytesTransferred = 0
            while (buffersCompleted < buffersPerAcquisition and not
                   ats.enter_pressed()):
                if triggering_instrument:
                    triggering_instrument.trigger()
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buffer = buffers[buffersCompleted % len(buffers)]
                t1 = time.time()
                self.board.waitAsyncBufferComplete(buffer.addr, timeout_ms=int(acquisition_timeout))
                t2 = time.time()
                buffersCompleted += 1
                bytesTransferred += buffer.size_bytes
    
                # TODO: Process sample data in this buffer. Data is available
                # as a NumPy array at buffer.buffer
    
                # NOTE:
                #
                # While you are processing this buffer, the board is already
                # filling the next available buffer(s).
                #
                # You MUST finish processing this buffer and post it back to the
                # board before the board fills all of its available DMA buffers
                # and on-board memory.
                #
                # Samples are arranged in the buffer as follows:
                # S0A, S0B, ..., S1A, S1B, ...
                # with SXY the sample number X of channel Y.
                #
                #
                # Sample codes are unsigned by default. As a result:
                # - 0x0000 represents a negative full scale input signal.
                # - 0x8000 represents a ~0V signal.
                # - 0xFFFF represents a positive full scale input signal.
                voltage_data = buffer.buffer.astype(float)
                if channel_str == 'A':
                    # copying contents of the buffer to my data variable
                    voltage_data = voltage_data[0:-excess_samples]
                    # converting from binary level to volts
                    voltage_data = self.channel_A_input_range*((voltage_data-32768)/32768)
                    # getting time axis
                    time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
                elif channel_str == 'B':
                    # copying contents of the buffer to my data variable
                    voltage_data = voltage_data[0:-excess_samples]
                    # converting from binary level to volts
                    voltage_data = self.channel_B_input_range*((voltage_data-32768)/32768)
                    # getting time axis
                    time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
                elif channel_str == 'AB':
                    #splitting it into A and B data
                    voltage_data_A = voltage_data[0:(len(voltage_data)//2)-excess_samples]
                    voltage_data_B = voltage_data[len(voltage_data)//2:-excess_samples]
                    # converting from binary level to volts
                    voltage_data_A = self.channel_A_input_range*((voltage_data_A-32768)/32768)
                    voltage_data_B = self.channel_B_input_range*((voltage_data_B-32768)/32768)
                    # getting time axis
                    time_axis = np.linspace(0,len(voltage_data_A)/self.sample_rate,len(voltage_data_A))

                # Add the buffer to the end of the list of available buffers.
                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        finally:
            print('FINISHED!')
            t3 = time.time()
            self.board.abortAsyncRead()
            t4 = time.time()
        # Compute the total transfer time, and display performance information.
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        if benchmark:
            buffersPerSec = 0
            bytesPerSec = 0
            recordsPerSec = 0
            if transferTime_sec > 0:
                buffersPerSec = buffersCompleted / transferTime_sec
                bytesPerSec = bytesTransferred / transferTime_sec
                recordsPerSec = recordsPerBuffer * buffersCompleted / transferTime_sec
            print("Captured %d buffers (%f buffers per sec)" %
                  (buffersCompleted, buffersPerSec))
            print("Captured %d records (%f records per sec)" %
                  (recordsPerBuffer * buffersCompleted, recordsPerSec))
            print("Transferred %d bytes (%f bytes per sec)" %
                  (bytesTransferred, bytesPerSec))
        
            print('Start to Wait: '+str(t1-t0))
            print('Wait time: '+str(t2-t1))
            print('Finish of wait to the end: '+str(t3-t2))
            print('Final step time: '+str(t4-t3))
            
            print('Transfer time - the rest = '+str(transferTime_sec-t4+t0))
        
        if channel_str == 'AB':
            return time_axis, voltage_data_A, voltage_data_B
        else:
            return time_axis,voltage_data
        
    def acquire_NPT_average(self, 
                record_length = 10.0E-6, # in seconds
                n_records = 100, # number of records
                triggering_instrument = None, # instrument that sends trigger to start acquisition
                channel_str = 'A', # 'A', 'B', or 'AB':
                acquisition_timeout = 15.0E3, # timeout in ms
                excess_samples = 45, #extra samples to throw out
                benchmark = False,
                n_buffers = None):
        #EXCESS SAMPLES REASON:
        #   often there is a saturation of the digitizer right at the end of
        #   an acquisition, so by default I acquire excess samples and throw
        #   them out at the end.
        #   * This will obviously not be suitable for all applications
        
        # NPT = no pre-trigger samples
        preTriggerSamples = 0
        postTriggerSamples = math.ceil(record_length*self.sample_rate) + excess_samples
        
        # Single acquisition: 1 record, 1 buffer
        recordsPerBuffer = 1
        buffersPerAcquisition = n_records
        
        if isinstance(n_buffers,type(None)):
            n_buffers = int(min(n_records,20))
                
        # TODO: Select the active channels.
        if channel_str == 'A':
            channels = ats.CHANNEL_A
        elif channel_str == 'B':
            channels = ats.CHANNEL_B
        elif channel_str == 'AB':
            channels = ats.CHANNEL_A | ats.CHANNEL_B
        else:
            sys.exit('ERROR: Invalid channel string')

        channelCount = 0
        for c in ats.channels:
            channelCount += (c & channels == c)
        
        # Compute the number of bytes per record and per buffer
        memorySize_samples, bitsPerSample = self.board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        samplesPerRecord = preTriggerSamples + postTriggerSamples
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount
            
        sample_type = ctypes.c_uint8
        if bytesPerSample > 1:
            sample_type = ctypes.c_uint16
            
        buffers = []
        for i in range(n_buffers):
            buffers.append(ats.DMABuffer(self.board.handle, sample_type, \
                                         bytesPerBuffer))
        
        # Set the record size
        self.board.setRecordSize(preTriggerSamples, postTriggerSamples)
    
        recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition
        
        # Configure the board to make an NPT AutoDMA acquisition
        self.board.beforeAsyncRead(channels,
                              -preTriggerSamples,
                              samplesPerRecord,
                              recordsPerBuffer,
                              recordsPerAcquisition,
                              ats.ADMA_EXTERNAL_STARTCAPTURE)
        
        # Post DMA buffers to board
        for buffer in buffers:
            self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        
        start = time.clock() # Keep track of when acquisition starteda
        try:
            self.board.startCapture() # Start the acquisition
            t0 = time.time()
            print("Capturing %d buffers." %
                  buffersPerAcquisition)
            buffersCompleted = 0
            bytesTransferred = 0
            while (buffersCompleted < buffersPerAcquisition and not
                   ats.enter_pressed()):
                if triggering_instrument:
                    pass
                    #triggering_instrument.trigger()
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buffer = buffers[buffersCompleted % len(buffers)]
                t1 = time.time()
                self.board.waitAsyncBufferComplete(buffer.addr, timeout_ms=int(acquisition_timeout))
                t2 = time.time()
                buffersCompleted += 1
                bytesTransferred += buffer.size_bytes
    
                # TODO: Process sample data in this buffer. Data is available
                # as a NumPy array at buffer.buffer
    
                # NOTE:
                #
                # While you are processing this buffer, the board is already
                # filling the next available buffer(s).
                #
                # You MUST finish processing this buffer and post it back to the
                # board before the board fills all of its available DMA buffers
                # and on-board memory.
                #
                # Samples are arranged in the buffer as follows:
                # S0A, S0B, ..., S1A, S1B, ...
                # with SXY the sample number X of channel Y.
                #
                #
                # Sample codes are unsigned by default. As a result:
                # - 0x0000 represents a negative full scale input signal.
                # - 0x8000 represents a ~0V signal.
                # - 0xFFFF represents a positive full scale input signal.
                new_voltage_data = buffer.buffer.astype(float)
                if buffersCompleted == 1:
                    voltage_data = new_voltage_data
                else:
                    voltage_data = (new_voltage_data + ((buffersCompleted-1)*voltage_data))/buffersCompleted

                # Add the buffer to the end of the list of available buffers.
                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        except Exception as e:
            print(e)
            print('Buffers Completed: '+str(buffersCompleted))
            sys.exit()
        finally:
            t3 = time.time()
            self.board.abortAsyncRead()
            t4 = time.time()
            
            if channel_str == 'A':
                # copying contents of the buffer to my data variable
                voltage_data = voltage_data[0:-excess_samples]
                # converting from binary level to volts
                voltage_data = self.channel_A_input_range*((voltage_data-32768)/32768)
                # getting time axis
                time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
            elif channel_str == 'B':
                # copying contents of the buffer to my data variable
                voltage_data = voltage_data[0:-excess_samples]
                # converting from binary level to volts
                voltage_data = self.channel_B_input_range*((voltage_data-32768)/32768)
                # getting time axis
                time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
            elif channel_str == 'AB':
                #splitting it into A and B data
                voltage_data_A = voltage_data[0:(len(voltage_data)//2)-excess_samples]
                voltage_data_B = voltage_data[len(voltage_data)//2:-excess_samples]
                # converting from binary level to volts
                voltage_data_A = self.channel_A_input_range*((voltage_data_A-32768)/32768)
                voltage_data_B = self.channel_B_input_range*((voltage_data_B-32768)/32768)
                # getting time axis
                time_axis = np.linspace(0,len(voltage_data_A)/self.sample_rate,len(voltage_data_A))
            
        # Compute the total transfer time, and display performance information.
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        if benchmark:
            buffersPerSec = 0
            bytesPerSec = 0
            recordsPerSec = 0
            if transferTime_sec > 0:
                buffersPerSec = buffersCompleted / transferTime_sec
                bytesPerSec = bytesTransferred / transferTime_sec
                recordsPerSec = recordsPerBuffer * buffersCompleted / transferTime_sec
            print("Captured %d buffers (%f buffers per sec)" %
                  (buffersCompleted, buffersPerSec))
            print("Captured %d records (%f records per sec)" %
                  (recordsPerBuffer * buffersCompleted, recordsPerSec))
            print("Transferred %d bytes (%f bytes per sec)" %
                  (bytesTransferred, bytesPerSec))
        
            print('Start to Wait: '+str(t1-t0))
            print('Wait time: '+str(t2-t1))
            print('Finish of wait to the end: '+str(t3-t2))
            print('Final step time: '+str(t4-t3))
            
            print('Transfer time - the rest = '+str(transferTime_sec-t4+t0))
        
        if channel_str == 'AB':
            return time_axis, voltage_data_A, voltage_data_B
        else:
            return time_axis,voltage_data
        
        
    def acquire_NPT_average_amplitude(self, 
                if_freq, 
                record_length = 10.0E-6, # in seconds
                n_records = 100, # number of records
                triggering_instrument = None, # instrument that sends trigger to start acquisition
                channel_str = 'A', # 'A', 'B', or 'AB':
                acquisition_timeout = 15.0E3, # timeout in ms
                excess_samples = 45, #extra samples to throw out
                benchmark = False):
        # DOESNT CURRENTLY SUPPORT TWO-CHANNEL OPERATION!!!
        # EXCESS SAMPLES REASON:
        #   often there is a saturation of the digitizer right at the end of
        #   an acquisition, so by default I acquire excess samples and throw
        #   them out at the end.
        #   * This will obviously not be suitable for all applications
        
        # NPT = no pre-trigger samples
        preTriggerSamples = 0
        postTriggerSamples = math.ceil(record_length*self.sample_rate) + excess_samples
        
        # Single acquisition: 1 record, 1 buffer
        recordsPerBuffer = 1
        buffersPerAcquisition = n_records
        n_buffers = int(min(n_records,30))
                
        # TODO: Select the active channels.
        if channel_str == 'A':
            channels = ats.CHANNEL_A
        elif channel_str == 'B':
            channels = ats.CHANNEL_B
        elif channel_str == 'AB':
            channels = ats.CHANNEL_A | ats.CHANNEL_B
        else:
            sys.exit('ERROR: Invalid channel string')

        channelCount = 0
        for c in ats.channels:
            channelCount += (c & channels == c)
        
        # Compute the number of bytes per record and per buffer
        memorySize_samples, bitsPerSample = self.board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        samplesPerRecord = preTriggerSamples + postTriggerSamples
        bytesPerRecord = bytesPerSample * samplesPerRecord
        bytesPerBuffer = bytesPerRecord * recordsPerBuffer * channelCount
            
        sample_type = ctypes.c_uint8
        if bytesPerSample > 1:
            sample_type = ctypes.c_uint16
            
        buffers = []
        for i in range(n_buffers):
            buffers.append(ats.DMABuffer(self.board.handle, sample_type, \
                                         bytesPerBuffer))
        
        # Set the record size
        self.board.setRecordSize(preTriggerSamples, postTriggerSamples)
    
        recordsPerAcquisition = recordsPerBuffer * buffersPerAcquisition
        
        # Configure the board to make an NPT AutoDMA acquisition
        self.board.beforeAsyncRead(channels,
                              -preTriggerSamples,
                              samplesPerRecord,
                              recordsPerBuffer,
                              recordsPerAcquisition,
                              ats.ADMA_EXTERNAL_STARTCAPTURE)
        
        # Post DMA buffers to board
        for buffer in buffers:
            self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        
        start = time.clock() # Keep track of when acquisition starteda
        try:
            self.board.startCapture() # Start the acquisition
            t0 = time.time()
            print("Capturing %d buffers." %
                  buffersPerAcquisition)
            buffersCompleted = 0
            bytesTransferred = 0
            while (buffersCompleted < buffersPerAcquisition and not
                   ats.enter_pressed()):
                if triggering_instrument:
                    pass
                    #triggering_instrument.trigger()
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buffer = buffers[buffersCompleted % len(buffers)]
                t1 = time.time()
                self.board.waitAsyncBufferComplete(buffer.addr, timeout_ms=int(acquisition_timeout))
                t2 = time.time()
                buffersCompleted += 1
                bytesTransferred += buffer.size_bytes
    
                # TODO: Process sample data in this buffer. Data is available
                # as a NumPy array at buffer.buffer
    
                # NOTE:
                #
                # While you are processing this buffer, the board is already
                # filling the next available buffer(s).
                #
                # You MUST finish processing this buffer and post it back to the
                # board before the board fills all of its available DMA buffers
                # and on-board memory.
                #
                # Samples are arranged in the buffer as follows:
                # S0A, S0B, ..., S1A, S1B, ...
                # with SXY the sample number X of channel Y.
                #
                #
                # Sample codes are unsigned by default. As a result:
                # - 0x0000 represents a negative full scale input signal.
                # - 0x8000 represents a ~0V signal.
                # - 0xFFFF represents a positive full scale input signal.
                voltage_data = buffer.buffer.astype(float)[0:-excess_samples]
                if channel_str == 'A':
                    voltage_data = self.channel_A_input_range*((voltage_data-32768)/32768)
                elif channel_str == 'B':
                    voltage_data = self.channel_B_input_range*((voltage_data-32768)/32768)
                else:
                    print('THIS FUNCTION DOESNT CURRENTLY SUPPORT TWO-CHANNEL OPERATION!')
                    sys.exit()
                    
                if buffersCompleted == 1:
                    #print('First Buffer Completed!')
                    if channel_str == 'A':
                        # getting time axis
                        time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
                    elif channel_str == 'B':
                        # getting time axis
                        time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
                    else:
                        print('THIS FUNCTION DOESNT CURRENTLY SUPPORT TWO-CHANNEL OPERATION!')
                        sys.exit()
                    inphase,quadrature = iq.get_iq(time_axis, voltage_data, if_freq, return_IQ = True)
                    amplitude_data = np.sqrt((inphase**2)+(quadrature**2))
                else:
                    #print('Buffer '+str(buffersCompleted)+' completed!')
                    inphase,quadrature = iq.get_iq(time_axis, voltage_data, if_freq, return_IQ = True)
                    new_amplitude_data = np.sqrt((inphase**2)+(quadrature**2))
                    amplitude_data = (new_amplitude_data + ((buffersCompleted-1)*amplitude_data))/buffersCompleted

                # Add the buffer to the end of the list of available buffers.
                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        except Exception as e:
            print(e)
            print('Buffers Completed: '+str(buffersCompleted))
            sys.exit()
        finally:
            t3 = time.time()
            self.board.abortAsyncRead()
            t4 = time.time()
            
            truncated_time_axis = time_axis[0:len(amplitude_data)]

        # Compute the total transfer time, and display performance information.
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        if benchmark:
            buffersPerSec = 0
            bytesPerSec = 0
            recordsPerSec = 0
            if transferTime_sec > 0:
                buffersPerSec = buffersCompleted / transferTime_sec
                bytesPerSec = bytesTransferred / transferTime_sec
                recordsPerSec = recordsPerBuffer * buffersCompleted / transferTime_sec
            print("Captured %d buffers (%f buffers per sec)" %
                  (buffersCompleted, buffersPerSec))
            print("Captured %d records (%f records per sec)" %
                  (recordsPerBuffer * buffersCompleted, recordsPerSec))
            print("Transferred %d bytes (%f bytes per sec)" %
                  (bytesTransferred, bytesPerSec))
        
            print('Start to Wait: '+str(t1-t0))
            print('Wait time: '+str(t2-t1))
            print('Finish of wait to the end: '+str(t3-t2))
            print('Final step time: '+str(t4-t3))
            
            print('Transfer time - the rest = '+str(transferTime_sec-t4+t0))
        
        if channel_str == 'AB':
            print('THIS FUNCTION DOESNT CURRENTLY SUPPORT TWO-CHANNEL OPERATION!')
            sys.exit()
        else:
            return truncated_time_axis,amplitude_data
        
    def acquire_TS_single(self, 
                acquisition_length = 10.0E-6, # in seconds
                triggering_instrument = None, # instrument that sends trigger to start acquisition
                channel_str = 'A', # 'A', 'B', or 'AB':
                acquisition_timeout = 15.0E3, # timeout in ms
                filename = None, # if a filename is specified, will save to file
                benchmark = False):
        #EXCESS SAMPLES REASON:
        #   often there is a saturation of the digitizer right at the end of
        #   an acquisition, so by default I acquire excess samples and throw
        #   them out at the end.
        #   * This will obviously not be suitable for all applications
        
        # TS = Triggered Streaming (Long dataset)
        
        # TODO: Select the total acquisition length in seconds
        acquisitionLength_sec = acquisition_length
    
        # TODO: Select the number of samples in each DMA buffer
        samplesPerBuffer = int(1.0E6)
        
        # TODO: Select the active channels.
        if channel_str == 'A':
            channels = ats.CHANNEL_A
        elif channel_str == 'B':
            channels = ats.CHANNEL_B
        elif channel_str == 'AB':
            channels = ats.CHANNEL_A | ats.CHANNEL_B
        else:
            sys.exit('ERROR: Invalid channel string')

        channelCount = 0
        for c in ats.channels:
            channelCount += (c & channels == c)
    
    
        # Compute the number of bytes per record and per buffer
        memorySize_samples, bitsPerSample = self.board.getChannelInfo()
        bytesPerSample = (bitsPerSample.value + 7) // 8
        bytesPerBuffer = bytesPerSample * samplesPerBuffer * channelCount;
        # Calculate the number of buffers in the acquisition
        samplesPerAcquisition = int(self.sample_rate * acquisitionLength_sec + 0.5);
        buffersPerAcquisition = ((samplesPerAcquisition + samplesPerBuffer - 1) //
                                 samplesPerBuffer)
    
        # TODO: Select number of DMA buffers to allocate
        bufferCount = 30
    
        # Allocate DMA buffers
    
        sample_type = ctypes.c_uint8
        if bytesPerSample > 1:
            sample_type = ctypes.c_uint16
    
        buffers = []
        for i in range(bufferCount):
            buffers.append(ats.DMABuffer(self.board.handle, sample_type, bytesPerBuffer))
        
        self.board.beforeAsyncRead(channels,
                              0,                 # Must be 0
                              samplesPerBuffer,
                              1,                 # Must be 1
                              0x7FFFFFFF,        # Ignored
                              ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_TRIGGERED_STREAMING)
        
    
    
        # Post DMA buffers to board
        for buffer in buffers:
            self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
    
        start = time.clock() # Keep track of when acquisition started
        try:
            self.board.startCapture() # Start the acquisition
            print("Capturing %d buffers." %
                  buffersPerAcquisition)
            buffersCompleted = 0
            bytesTransferred = 0
            while (buffersCompleted < buffersPerAcquisition and not
                   ats.enter_pressed()):
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buffer = buffers[buffersCompleted % len(buffers)]
                self.board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
                buffersCompleted += 1
                bytesTransferred += buffer.size_bytes
    
                # TODO: Process sample data in this buffer. Data is available
                # as a NumPy array at buffer.buffer
    
                # NOTE:
                #
                # While you are processing this buffer, the board is already
                # filling the next available buffer(s).
                #
                # You MUST finish processing this buffer and post it back to the
                # board before the board fills all of its available DMA buffers
                # and on-board memory.
                #
                # Samples are arranged in the buffer as follows:
                # S0A, S0B, ..., S1A, S1B, ...
                # with SXY the sample number X of channel Y.
                #
                #
                # Sample codes are unsigned by default. As a result:
                # - 0x0000 represents a negative full scale input signal.
                # - 0x8000 represents a ~0V signal.
                # - 0xFFFF represents a positive full scale input signal.
                # Optionaly save data to file
                new_voltage_data = buffer.buffer.astype(float)
                if buffersCompleted == 1:
                    if channel_str == 'AB':
                        voltage_data_A = new_voltage_data[0:(len(new_voltage_data)//2)]
                        voltage_data_B = new_voltage_data[len(new_voltage_data)//2:]
                    else:
                        voltage_data = new_voltage_data
                else:
                    if channel_str == 'AB':
                        voltage_data_A = np.hstack((voltage_data_A,new_voltage_data[0:(len(new_voltage_data)//2)]))
                        voltage_data_B = np.hstack((voltage_data_B,new_voltage_data[len(new_voltage_data)//2:]))
                    else:
                        voltage_data = np.hstack((voltage_data,new_voltage_data))
    
                # Add the buffer to the end of the list of available buffers.
                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        finally:
            self.board.abortAsyncRead()
            if buffersCompleted==0:
                print('ERROR: NO BUFFERS COMPLETED, CHECK TRIGGER')
            if channel_str == 'A':
                # converting from binary level to volts
                voltage_data = self.channel_A_input_range*((voltage_data-32768)/32768)
                # getting time axis
                time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
            elif channel_str == 'B':
                # converting from binary level to volts
                voltage_data = self.channel_B_input_range*((voltage_data-32768)/32768)
                # getting time axis
                time_axis = np.linspace(0,len(voltage_data)/self.sample_rate,len(voltage_data))
            elif channel_str == 'AB':
                # converting from binary level to volts
                voltage_data_A = self.channel_A_input_range*((voltage_data_A-32768)/32768)
                voltage_data_B = self.channel_B_input_range*((voltage_data_B-32768)/32768)
                # getting time axis
                time_axis = np.linspace(0,len(voltage_data_A)/self.sample_rate,len(voltage_data_A))
                
        # Compute the total transfer time, and display performance information.
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
        buffersPerSec = 0
        bytesPerSec = 0
        if transferTime_sec > 0:
            buffersPerSec = buffersCompleted / transferTime_sec
            bytesPerSec = bytesTransferred / transferTime_sec
        print("Captured %d buffers (%f buffers per sec)" %
              (buffersCompleted, buffersPerSec))
        print("Transferred %d bytes (%f bytes per sec)" %
              (bytesTransferred, bytesPerSec))
    
        if channel_str == 'AB':
            return time_axis, voltage_data_A, voltage_data_B
        else:
            return time_axis,voltage_data
        
        
        
        