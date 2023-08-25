# uncompyle6 version 3.2.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.0 (default, Jun 28 2018, 08:04:48) [MSC v.1912 64 bit (AMD64)]
# Embedded file name: C:\Users\Ben\Documents\python_code\modules\instrument_classes_module.py
# Size of source mod 2**32: 69357 bytes
"""
Created on Wed Nov 14 16:36:20 2018

@author: Ben
"""
import numpy as np, struct, visa, nidaqmx, statistics

class gpib_instrument:

    def __init__(self, addr):
        addr_str = 'GPIB0::' + str(addr) + '::INSTR'
        self.instr = visa.ResourceManager().open_resource(addr_str)

    def write(self, message, return_output=0):
        if return_output:
            return self.instr.write(message)
        self.instr.write(message)

    def query(self, message):
        return self.instr.query(message)

    def write_raw(self, message, return_output=0):
        if return_output:
            return self.instr.write_raw(message)
        self.instr.write_raw(message)

    def set_timeout(self, timeout):
        self.instr.timeout = timeout


class keysight_n5183b(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def set_frequency(self, freq):
        message = ':freq ' + str(freq)
        self.write(message)

    def set_power(self, power, units='dbm'):
        message = ':pow ' + str(power) + units
        self.write(message)

    def set_phase(self, phase, units='rad'):
        message = ':phas ' + str(phase) + units
        self.write(message)

    def toggle_output(self, state):
        message = ':outp ' + str(state)
        self.write(message)

    def toggle_modulation(self, state):
        message = ':outp:mod ' + str(state)
        self.write(message)

    def toggle_pulse_mode(self, state):
        message = ':pulm:stat ' + str(state)
        self.write(message)

    def toggle_alc(self, state):
        message = ':pow:alc ' + str(state)
        self.write(message)

    def set_pulse_source(self, source):
        if source == 'ext':
            message = 'pulm:sour ' + source
        else:
            message = 'pulm:sour:int ' + source
        self.write(message)

    def set_pulse_delay(self, delay, units='s'):
        message = ':pulm:int:del ' + str(delay) + units
        self.write(message)

    def set_pulse_width(self, width, units='s'):
        message = ':pulm:int:pwid ' + str(width) + units
        self.write(message)


class agilent_e8257c(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def toggle_output(self, state):
        message = ':outp ' + str(state)
        self.write(message)

    def toggle_modulation(self, state):
        message = ':outp:mod ' + str(state)
        self.write(message)

    def set_frequency(self, freq, units='Hz'):
        message = ':freq ' + str(freq) + units
        self.write(message)

    def set_phase(self, phase, units='rad'):
        message = ':phas ' + str(phase) + units
        self.write(message)

    def toggle_alc(self, state):
        message = ':pow:alc ' + str(state)
        self.write(message)

    def set_power(self, power, units='dbm'):
        message = ':pow ' + str(power) + units
        self.write(message)

    def toggle_pulse_mode(self, state):
        message = ':pulm:stat ' + str(state)
        self.write(message)

    def set_pulse_source(self, source):
        if source == 'ext':
            message = 'pulm:sour ' + source
        else:
            message = 'pulm:sour:int ' + source
        self.write(message)

    def set_pulse_delay(self, delay, units='s'):
        message = ':pulm:int:del ' + str(delay) + units
        self.write(message)

    def set_pulse_width(self, width, units='s'):
        message = ':pulm:int:pwid ' + str(width) + units
        self.write(message)


class hp_83711b(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def set_frequency(self, freq, units='Hz'):
        message = 'freq ' + str(freq) + units
        self.write(message)

    def set_power(self, power, units='dBm'):
        message = 'pow ' + str(power) + units
        self.write(message)

    def toggle_output(self, state):
        message = 'outp ' + str(state)
        self.write(message)


class hp_8648c(gpib_instrument):

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


class hp_34401a(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def get_voltage(self):
        message = 'meas:volt:dc?'
        return float(self.query(message))


class agilent_e3634a(gpib_instrument):

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


class agilent_e4404b(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def abort(self):
        message = ':abor'
        self.write(message)

    def force_trigger(self):
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


class agilent_e4408b(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def abort(self):
        message = ':abor'
        self.write(message)

    def force_trigger(self):
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


class agilent_e5071c(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def allocate_channels(self, alloc_str):
        message = ':disp:spl D' + alloc_str
        self.write(message)

    def set_channel(self, channel):
        message = ':disp:wind' + str(channel) + ':act'
        self.write(message)

    def query_channel(self):
        message = ':serv:chan:act?'
        channel = self.query(message)
        return int(channel)

    def set_num_traces(self, num_traces, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':calc' + str(channel) + ':par:coun ' + str(num_traces)
        self.write(message)

    def query_num_traces(self, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':calc' + str(channel) + ':par:coun?'
        num_traces = self.query(message)
        return int(num_traces)

    def allocate_traces(self, alloc_str, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':disp:wind' + str(channel) + ':spl D' + alloc_str
        self.write(message)

    def set_trace(self, trace, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':calc' + str(channel) + ':par' + str(trace) + ':sel'
        self.write(message)

    def query_trace(self, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':serv:chan' + str(channel) + ':trac:act?'
        trace = self.query(message)
        return int(trace)

    def autoscale(self, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':disp:wind' + str(channel) + ':trac' + str(trace) + ':y:auto'
        self.write(message)

    def toggle_marker(self, marker, state, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':calc' + str(channel) + ':mark' + str(marker) + ' ' + str(state)
        self.write(message)

    def move_marker(self, marker, freq, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':calc' + str(channel) + ':mark' + str(marker) + ':x ' + str(freq)
        self.write(message)

    def activate_marker(self, marker, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':calc' + str(channel) + ':mark' + str(marker) + ':act'
        self.write(message)

    def marker_search(self, marker, type_str, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark' + str(marker) + ':func'
        self.write(message + ':type ' + type_str)
        self.write(message + ':exec')

    def marker_track(self, marker, type_str, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark' + str(marker) + ':func'
        self.write(message + ':type ' + type_str)
        self.toggle_marker_tracking(marker, 1, channel, trace)

    def toggle_marker_tracking(self, marker, state, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark' + str(marker) + ':func:trac ' + str(state)
        self.write(message)

    def toggle_marker_search_range(self, state, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark:func:dom ' + str(state)
        self.write(message)

    def set_marker_search_start(self, freq, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark:func:dom:star ' + str(freq)
        self.write(message)

    def set_marker_search_stop(self, freq, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark:func:dom:stop ' + str(freq)
        self.write(message)

    def toggle_bandwidth_search(self, state, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark:bwid ' + str(state)
        self.write(message)

    def set_bandwidth_threshold(self, marker, threshold, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':mark' + str(marker) + ':bwid:thr ' + str(threshold)
        self.write(message)

    def track_resonance(self, marker=1, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        self.toggle_bandwidth_search(1, channel, trace)
        self.set_bandwidth_threshold(marker, 3, channel, trace)
        self.toggle_marker(marker, 1, channel)
        self.activate_marker(marker, channel)
        self.marker_track(marker, 'min', channel, trace)

    def toggle_output(self, state):
        message = ':outp ' + str(state)
        self.write(message)

    def set_measurement(self, meas_str, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':par' + str(trace) + ':def ' + meas_str
        self.write(message)

    def query_measurement(self, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace()
        message = ':calc' + str(channel) + ':par' + str(trace) + ':def?'
        meas_str = self.query(message)
        return meas_str[:-1]

    def set_sweep_type(self, type_str, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':swe:type ' + type_str
        self.write(message)

    def query_sweep_type(self, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':swe:type?'
        type_str = self.query(message)
        return type_str[:-1]

    def set_sweep_mode(self, mode_str, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':swe:gen ' + mode_str
        self.write(message)

    def query_sweep_mode(self, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':swe:gen?'
        mode_str = self.query(message)
        return mode_str[:-1]

    def set_sweep_points(self, points, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':swe:poin ' + str(points)
        self.write(message)

    def set_frequency_start(self, freq, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':freq:star ' + str(freq)
        self.write(message)

    def set_frequency_stop(self, freq, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':freq:stop ' + str(freq)
        self.write(message)

    def set_frequency_center(self, freq, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':freq:cent ' + str(freq)
        self.write(message)

    def set_frequency_span(self, freq, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':freq:span ' + str(freq)
        self.write(message)

    def set_freqs(self, freq1, freq2, interval_type='range', channel=None):
        if not channel:
            channel = self.query_channel()
        if interval_type == 'range':
            self.set_frequency_start(freq1, channel)
            self.set_frequency_stop(freq2, channel)
        else:
            if interval_type == 'span':
                self.set_frequency_center(freq1, channel)
                self.set_frequency_span(freq2, channel)
            else:
                print('ERROR: Invalid Interval Type!')

    def set_IF_bandwidth(self, bandwidth, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':band ' + str(bandwidth)
        self.write(message)

    def set_power(self, power, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sour' + str(channel) + ':pow ' + str(power)
        self.write(message)

    def set_format(self, format_str, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':form ' + format_str
        self.write(message)

    def query_format(self, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':form?'
        format_str = self.query(message)
        return format_str[:-1]

    def set_electrical_delay(self, delay, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':corr:edel:time ' + str(delay)
        self.write(message)

    def toggle_averaging(self, state, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':aver ' + str(state)
        self.write(message)

    def set_averaging(self, factor, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':aver:coun ' + str(factor)
        self.write(message)

    def restart_averaging(self, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':aver:cle'
        self.write(message)

    def toggle_averaging_trigger(self, state):
        message = ':trig:aver ' + str(state)
        self.write(message)

    def toggle_continuous_triggering(self, state, channel=None):
        if not channel:
            channel = self.query_channel()
        message = ':init' + str(channel) + ':cont ' + str(state)
        self.write(message)

    def set_trigger_source(self, source_str):
        message = ':trig:sour ' + source_str
        self.write(message)

    def set_trigger_scope(self, scope_str):
        message = ':trig:scop ' + scope_str
        self.write(message)

    def trigger(self, wait=True):
        message = ':trig:sing'
        self.write(message)
        if wait:
            return self.query('*OPC?')

    def transfer_data_to_memory(self, channel=None, trace=None):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':math:mem'
        self.write(message)

    def get_trace_data(self, channel=None, trace=None, convert=True):
        if not channel:
            channel = self.query_channel()
        if not trace:
            trace = self.query_trace(channel)
        message = ':calc' + str(channel) + ':trac' + str(trace) + ':data:fmem?'
        data_str = self.query(message)
        format_str = self.query_format(channel, trace)
        data_list = data_str.split(',')
        return_list = []
        for ii in range(len(data_list) // 2):
            if format_str == 'SMIT':
                resistance = float(data_list[2 * ii])
                reactance = float(data_list[2 * ii + 1])
                if convert:
                    return_list.append(complex(resistance, reactance))
                if not convert:
                    return_list.append(str(complex(resistance, reactance)))
                    return_str = ','.join(return_list)
            elif convert:
                return_list.append(float(data_list[2 * ii]))
            else:
                return_list.append(data_list[2 * ii])
                return_str = ','.join(return_list)

        if convert:
            return return_list
        else:
            return return_str

    def get_frequency_data(self, channel=None, convert=True):
        if not channel:
            channel = self.query_channel()
        message = ':sens' + str(channel) + ':freq:data?'
        freq_str = self.query(message)
        if convert:
            freq_list = freq_str.split(',')
            freq_list = [float(ii) for ii in freq_list]
            return freq_list
        else:
            return freq_str[:-1]

    def get_parameters(self):
        pass


class tektronix_tds7704b(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def toggle_channel(self, channel, state):
        message = 'sel:ch' + str(channel) + ' ' + str(state)
        self.write(message)

    def set_coupling(self, channel, coupling_string):
        message = 'ch' + str(channel) + ':coup ' + coupling_string
        self.write(message)

    def set_vertical_offset(self, channel, offset):
        message = 'ch' + str(channel) + ':offs ' + str(offset)
        self.write(message)

    def set_vertical_position(self, channel, divisions):
        message = 'ch' + str(channel) + ':pos ' + str(divisions)
        self.write(message)

    def set_vertical_scale(self, channel, scale):
        message = 'ch' + str(channel) + ':sca ' + str(scale)
        self.write(message)


class tektronix_awg7052(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def force_trigger(self):
        message = '*trg'
        self.write(message)

    def toggle_output(self, channel, state):
        message = 'outp' + str(channel) + ' ' + str(state)
        self.write(message)

    def run(self):
        message = 'awgc:run'
        self.write(message)

    def stop(self):
        message = 'awgc:stop'
        self.write(message)

    def toggle_run(self, state):
        if state:
            self.run()
        else:
            self.stop()

    def set_sampling_rate(self, rate):
        message = 'sour:freq ' + str(rate)
        self.write(message)

    def set_run_mode(self, mode):
        message = 'awgc:rmode ' + mode
        self.write(message)

    def set_frequency_reference(self, source):
        message = 'sour:rosc:sour ' + source
        self.write(message)

    def new_waveform(self, name, size):
        message = 'wlis:wav:new "' + name + '"' + ',' + str(size) + ',REAL'
        self.write(message)

    def delete_waveform(self, name):
        message = 'wlis:wav:del "' + name + '"'
        self.write(message)

    def clear_waveforms(self):
        message = 'wlis:wav:del ALL'
        self.write(message)

    def send_waveform(self, name, w, m1, m2):
        n_samples = len(w)
        self.delete_waveform(name)
        self.new_waveform(name, n_samples)
        m = 128 * m2 + 64 * m1
        bytes_data = b''
        for ii in range(n_samples):
            bytes_data += struct.pack('fB', w[ii], int(m[ii]))

        num_bytes = n_samples * 5
        num_bytes = str(num_bytes)
        num_digits = str(len(num_bytes))
        num_bytes = num_bytes.encode('ascii')
        num_digits = num_digits.encode('ascii')
        bytes_count = num_digits + num_bytes
        bytes_name = name.encode('ascii')
        bytes_samples = str(n_samples)
        bytes_samples = bytes_samples.encode('ascii')
        message = b'wlis:wav:data "' + bytes_name + b'",0,' + bytes_samples + b',#' + bytes_count + bytes_data
        self.write_raw(message)

    def load_waveform(self, channel, name):
        message = 'sour' + str(channel) + ':wav "' + name + '"'
        self.write(message)

    def set_analog_amplitude(self, channel, amplitude, units='V'):
        message = 'sour' + str(channel) + ':volt ' + str(amplitude) + units
        self.write(message)

    def set_marker_low(self, channel, marker, voltage, units='V'):
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':volt:low ' + str(voltage) + units
        self.write(message)

    def set_marker_high(self, channel, marker, voltage, units='V'):
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':volt:high ' + str(voltage) + units
        self.write(message)

    def set_marker_delay(self, channel, marker, delay):
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':del ' + str(delay) + 'ps'
        self.write(message)

    def query_cwd(self):
        message = 'mmem:cdir?'
        return self.query(message)

    def mkdir(self, dir_name):
        message = 'mmem:mdir "' + dir_name + '"'
        self.write(message)

    def ls(self):
        return self.query('mmem:cat?')

    def cd(self, rel_path):
        message = 'mmem:cdir "' + rel_path + '"'
        self.write(message)

    def reset_cwd(self):
        message = 'mmem:cdir'
        self.write(message)

    def set_cwd(self, absolute_path):
        self.reset_cwd()
        message = 'mmem:cdir "' + absolute_path + '"'
        self.write(message)


class tektronix_awg520(gpib_instrument):

    def __init__(self, addr):
        super().__init__(addr)

    def force_trigger(self):
        message = '*trg'
        self.write(message)

    def toggle_output(self, channel, state):
        message = 'outp' + str(channel) + ' ' + str(state)
        self.write(message)

    def set_run_mode(self, mode):
        message = 'awgc:rmode ' + mode
        self.write(message)

    def run(self):
        message = 'awgc:run'
        self.write(message)

    def stop(self):
        message = 'awgc:stop'
        self.write(message)

    def toggle_run(self, state):
        if state:
            self.run()
        else:
            self.stop()

    def set_offset(self, channel, offset, units='V'):
        message = 'sour' + str(channel) + ':volt:offs ' + str(offset) + units
        self.write(message)

    def set_amplitude(self, channel, amplitude, units='V'):
        message = 'sour' + str(channel) + ':volt ' + str(amplitude) + units
        self.write(message)

    def set_frequency_reference(self, channel, source):
        message = 'sour' + str(channel) + ':rosc:sour ' + source
        self.write(message)

    def send_waveform(self, w, m1, m2, filename, samplerate):
        bytes_filename = filename.encode('ascii')
        cmd = b'mmem:data "' + bytes_filename + b'",'
        header = b'MAGIC 1000\r\n'
        nsamples = len(w)
        m = m1 + np.multiply(m2, 2)
        bytes_data = b''
        for ii in range(nsamples):
            bytes_data += struct.pack('<fB', w[ii], int(m[ii]))

        num_bytes = len(bytes_data)
        num_digits = len(str(num_bytes))
        num_digits = str(num_digits)
        num_digits = num_digits.encode('ascii')
        num_bytes = str(num_bytes)
        num_bytes = num_bytes.encode('ascii')
        data_counter = b'#' + num_digits + num_bytes
        samplerate_str = '{:.2E}'.format(samplerate)
        samplerate_bytes = samplerate_str.encode('ascii')
        trailer = b'CLOCK ' + samplerate_bytes + b'\r\n'
        file = header + data_counter + bytes_data + trailer
        num_file_bytes = len(file)
        num_file_digits = len(str(num_file_bytes))
        num_file_digits = str(num_file_digits)
        num_file_digits = num_file_digits.encode('ascii')
        num_file_bytes = str(num_file_bytes)
        num_file_bytes = num_file_bytes.encode('ascii')
        file_counter = b'#' + num_file_digits + num_file_bytes
        message = cmd + file_counter + file
        self.write_raw(message)

    def load_waveform(self, channel, filename):
        message = 'sour' + str(channel) + ':func:user ' + '"' + filename + '"'
        self.write(message)

    def set_marker_low(self, channel, marker, voltage, units='V'):
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':volt:low ' + str(voltage) + units
        self.write(message)

    def set_marker_high(self, channel, marker, voltage, units='V'):
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':volt:high ' + str(voltage) + units
        self.write(message)

    def set_marker_delay(self, channel, marker, delay, units='s'):
        message = 'sour' + str(channel) + ':mark' + str(marker) + ':del ' + str(delay) + units
        self.write(message)

    def set_mass_storage(self, device='MAIN'):
        message = 'mmem:msis "' + device + '"'
        self.write(message)

    def query_cwd(self):
        message = 'mmem:cdir?'
        return self.query(message)

    def mkdir(self, dir_name):
        message = 'mmem:mdir "' + dir_name + '"'
        self.write(message)

    def ls(self):
        return self.query('mmem:cat?')

    def cd(self, rel_path):
        message = 'mmem:cdir "' + rel_path + '"'
        self.write(message)

    def reset_cwd(self):
        message = 'mmem:cdir'
        self.write(message)

    def set_cwd(self, absolute_path):
        self.reset_cwd()
        message = 'mmem:cdir "' + absolute_path + '"'
        self.write(message)


class national_instruments_bnc2090:

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