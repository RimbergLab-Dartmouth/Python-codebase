# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:38:52 2018

@author: Ben
"""
import atsapi as ats
import time
import os
import math
import ctypes
import sys
import numpy as np
import tkinter as tk

class ats9462:
    def __init__(self, \
                 input_range = 4.0, \
                 BW_limit = 0, \
                 clock_source = 'ext', \
                 trigger_delay = 0):
        # input_range = float, |voltage| limit in volts
        #               can pass list of floats to set channels A,B separately
        # clock_source = str: 'int' or 'ext'
        # BW_limit = 0, 1, or [chA_flag,chB_flag]
        
        self.board = ats.Board(systemId = 1, boardId = 1)
        
        # internal: ats.INTERNAL_CLOCK
        #   - sample rate -> flag from atsapi
        # external ref: ats.EXTERNAL_CLOCK_10MHz_REF
        #   - sample rate -> int in Hz
        self.sample_rate = 180.0E6 # NEED TO CHANGE BOTH THIS AND THE ATS FLAG
        if clock_source == 'ext':
            clock_source_code = ats.EXTERNAL_CLOCK_10MHz_REF
            sample_rate_code = int(self.sample_rate)
        elif clock_source == 'int':
            clock_source_code = ats.INTERNAL_CLOCK
            sample_rate_code = ats.SAMPLE_RATE_180MSPS
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
        channel_A_coupling = ats.DC_COUPLING
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
        first_level = 128
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
    
        self.trigger_delay_samples = int(trigger_delay * self.sample_rate + 0.5)
    
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
                record_length = 1.0E-6, # in seconds
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
        buffer_count = 1
                
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
        for i in range(buffer_count):
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
            print("Capturing %d buffers. Press <enter> to abort" %
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
        
        
        