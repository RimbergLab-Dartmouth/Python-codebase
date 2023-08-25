# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:52:22 2018

@author: Ben
"""

import numpy as np
import math
import struct
import visa
import matplotlib
import matplotlib.pyplot as plt
#import gpib_module as gpib

#awg = gpib.connect(8)

filename = 'sine.wfm'
f = open(filename,'rb')
chunk_number = 0

#first the magic 1000\r\n
chunk = f.read(12)
chunk_number += 1
chunk = chunk.decode('utf-8')
if chunk == 'MAGIC 1000\r\n':
    print('Chunk #'+str(chunk_number)+' decoded successfully!\n')
else:
    print('ERROR: Chunk #'+str(chunk_number)+' unsuccessfully decoded\n')

#second bit of information is the number sign - just a delimiter really
chunk = f.read(1)
chunk_number += 1
chunk = chunk.decode('utf-8')
if chunk == '#':
    print('Chunk #'+str(chunk_number)+' decoded successfully!\n')
else:
    print('ERROR: Chunk #'+str(chunk_number)+' unsuccessfully decoded\n')


#third chunk contains the number of digits in the byte-count
chunk = f.read(1)
chunk_number += 1
num_digits = chunk.decode('utf-8')
num_digits = int(num_digits)


#fourth chunk is num_digits long and contains the number of bytes in the data
chunk = f.read(num_digits)
chunk_number += 1
num_bytes = chunk.decode('utf-8')
num_bytes = int(num_bytes)

print('number of bytes = '+str(num_bytes))

num_data_pts = num_bytes//5
waveform = []
markers = []

for ii in range(num_data_pts):
    chunk = f.read(5)
    data_chunk,marker_chunk = struct.unpack('<fB',chunk)
    waveform.append(data_chunk)
    markers.append(marker_chunk)



rest = f.read()
f.close()

print(rest)

plt.plot(waveform)
plt.plot(markers)
plt.show()


"""
print(type(data))
print(len(data))

print('\n first chunk:')
print(first_chunk)
print(type(first_chunk))
print(len(first_chunk))

print('\n second chunk:')
print(second_chunk)
print(type(second_chunk))
print(len(second_chunk))
"""