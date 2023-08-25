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

samplesPerSec = 180000000.0
def get_default_params():
    # internal: ats.INTERNAL_CLOCK
    #   - sample rate -> flag from atsapi
    # external ref: ats.EXTERNAL_CLOCK_10MHz_REF
    #   - sample rate -> int in Hz
    clock_source = ats.INTERNAL_CLOCK
    global samplesPerSec
    samplesPerSec = 180000000.0 # NEED TO CHANGE BOTH THIS AND THE ATS FLAG
    if clock_source == ats.EXTERNAL_CLOCK_10MHz_REF:
        sample_rate = int(samplesPerSec)
    elif clock_source == ats.INTERNAL_CLOCK:
        sample_rate = ats.SAMPLE_RATE_180MSPS
    else:
        print('Unexpected Clock Source!')
    clock_edge = ats.CLOCK_EDGE_RISING
    decimation = 0
    clock_params = (clock_source, sample_rate, clock_edge, decimation)

    channel_A_id = ats.CHANNEL_A
    channel_A_coupling = ats.AC_COUPLING
    channel_A_input_range = ats.INPUT_RANGE_PM_2_V
    channel_A_impedence = ats.IMPEDANCE_50_OHM
    channel_A_params = (channel_A_id, channel_A_coupling, \
                        channel_A_input_range, channel_A_impedence)

    channel_B_id = ats.CHANNEL_B
    channel_B_coupling = ats.DC_COUPLING
    channel_B_input_range = ats.INPUT_RANGE_PM_2_V
    channel_B_impedence = ats.IMPEDANCE_50_OHM
    channel_B_params = (channel_B_id, channel_B_coupling, \
                        channel_B_input_range, channel_B_impedence)
    
    channel_A_bandwidth = 0
    channel_B_bandwidth = 0

    trigger_engine_operation = ats.TRIG_ENGINE_OP_J
    first_engine = ats.TRIG_ENGINE_J
    first_source = ats.TRIG_EXTERNAL
    first_slope = ats.TRIGGER_SLOPE_POSITIVE
    first_level = 150
    second_engine = ats.TRIG_ENGINE_K
    second_source = ats.TRIG_DISABLE
    second_slope = ats.TRIGGER_SLOPE_POSITIVE
    second_level = 128
    trigger_params = (trigger_engine_operation, \
                      first_engine, first_source, first_slope, first_level, \
                      second_engine, second_source, second_slope, second_level)

    external_trigger_coupling = ats.DC_COUPLING
    external_trigger_input_range = ats.ETR_1V
    external_trigger_params = (external_trigger_coupling, \
                               external_trigger_input_range)

    triggerDelay_sec = 0
    trigger_delay = int(triggerDelay_sec * samplesPerSec + 0.5)

    triggerTimeout_sec = 0
    trigger_timeout = int(triggerTimeout_sec / 10e-6 + 0.5)
    
    aux_mode = ats.AUX_OUT_PACER
    aux_param = 18
    aux_io_params = (aux_mode, aux_param)
    
    return  clock_params, \
            channel_A_params, channel_A_bandwidth, \
            channel_B_params, channel_B_bandwidth, \
            trigger_params, external_trigger_params, \
            trigger_delay, trigger_timeout, \
            aux_io_params

class ats9462:
    def __init__(self):
        self.board = ats.Board(systemId = 1, boardId = 1)
        self.configure_default()
        
    def configure_default(self):
        clock_params, \
            channel_A_params, channel_A_bandwidth, \
            channel_B_params, channel_B_bandwidth, \
            trigger_params, external_trigger_params, \
            trigger_delay, trigger_timeout, \
            aux_io_params = get_default_params()

        self.board.setCaptureClock(*clock_params)
        self.board.inputControlEx(*channel_A_params)
        self.board.setBWLimit(ats.CHANNEL_A, channel_A_bandwidth)
        self.board.inputControlEx(*channel_B_params)
        self.board.setBWLimit(ats.CHANNEL_B, channel_B_bandwidth)
        self.board.setTriggerOperation(*trigger_params)
        self.board.setExternalTrigger(*external_trigger_params)
        self.board.setTriggerDelay(trigger_delay)
        self.board.setTriggerTimeOut(trigger_timeout)
        self.board.configureAuxIO(*aux_io_params)
        
    def acquire_NPT(self,
                preTriggerSamples = 0, 
                record_length = 10**-6, # in seconds
                recordsPerBuffer = 1, 
                buffersPerAcquisition = 1, 
                buffer_count = 1, # number of buffers to allocate
                channel_str = 'A', # 'A', 'B', or 'AB'
                save_data = True, 
                filename = None, 
                return_data = False):
        
        postTriggerSamples = math.ceil(record_length*samplesPerSec)
                
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

        # TODO: Should data be saved to file? YES
        if save_data:
            dataFile = open(os.path.join(os.getcwd(),
                                         filename), 'wb')
            #print('SAVING DATA')
            #print(dataFile)
            #print(filename)
        
        if return_data:
            data_list = []
        
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
                # Wait for the buffer at the head of the list of available
                # buffers to be filled by the board.
                buffer = buffers[buffersCompleted % len(buffers)]
                t1 = time.time()
                self.board.waitAsyncBufferComplete(buffer.addr, timeout_ms=5000)
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
                # Optionaly save data to file
                
                #print(buffer.buffer)
                
                if dataFile:
                    buffer.buffer.tofile(dataFile)
                    if return_data:
                        data_list.append(buffer.buffer.copy())
                        #print('Data list: \n')
                        #print(data_list)
    
                # Add the buffer to the end of the list of available buffers.
                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
        finally:
            t3 = time.time()
            self.board.abortAsyncRead()
            t4 = time.time()
        # Compute the total transfer time, and display performance information.
        transferTime_sec = time.clock() - start
        print("Capture completed in %f sec" % transferTime_sec)
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
        
        if return_data:
            return data_list
        
        
        