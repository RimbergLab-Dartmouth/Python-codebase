# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:27:53 2018

@author: Ben
"""

import instrument_classes_module as icm
import tkinter as tk

daq_board = icm.national_instruments_bnc2090()

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
    daq_board.set_voltage(0,voltage)

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
    daq_board.set_voltage(1,voltage)

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













