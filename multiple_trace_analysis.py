# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 22:50:41 2022

@author: Zhangyu Wang
"""

import os
import obspy
from obspy import read,UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'data'
st = obspy.Stream()
for sacname in os.listdir(data_dir):
    st = st + read(os.path.join(data_dir,sacname))
st.filter('bandpass', freqmin=5, freqmax=120)

#cut_time = UTCDateTime(2022,8,31,17,51,25)
#cut_time = cut_time-8*3600
#cut_start, cut_end = cut_time-12, cut_time+12
cut_time = UTCDateTime(2022,8,31,17,51,25)
cut_time = cut_time-8*3600
cut_start, cut_end = cut_time-12, cut_time+3*60
st_cut = st.slice(cut_start, cut_end)

# In[] plot waveform
plt.figure(figsize=(30,5))
ax = plt.subplot(111)
max_amp = np.max(np.abs(st_cut.max()))
trace_name = []
data = []
for i,tr in enumerate(st_cut):
    data_ = tr.data/max_amp   
    data.append(data_)
    ax.plot(data_+i,'k',linewidth=1)
    trace_name.append(tr.stats.station[4:])
ax.yaxis.set_ticks(np.arange(0,i+1, 1))
ax.set_yticklabels(trace_name)


data = np.array(data)

template = data[5,3500:4500]

#a = np.correlate(data[0,:],data[5,:],mode='same')

a = np.correlate(data[2,:],template,mode='same')
#plt.plot(a)
print(np.where(a==a.max()))


# In[]
cut_time = UTCDateTime(2022,8,31,17,51,25)
cut_time = cut_time-8*3600
cut_start, cut_end = cut_time-12, cut_time+10*60
st_cut = st.slice(cut_start, cut_end)

# In[] plot waveform
plt.figure(figsize=(15,5))
ax = plt.subplot(111)
max_amp = np.max(np.abs(st_cut.max()))
trace_name = []
data = []
for i,tr in enumerate(st_cut):
    data_ = tr.data/max_amp   
    data.append(data_)
    ax.plot(data_+i,'k',linewidth=1)
    trace_name.append(tr.stats.station[4:])
ax.yaxis.set_ticks(np.arange(0,i+1, 1))
ax.set_yticklabels(trace_name)

data = np.array(data)

def data2energy(d):
    k=125
    npts = len(d)
    e = np.zeros((npts))
    for i,amp in enumerate(d):
        if i>125 and i<npts-k:
            e[i] = np.sum(d[i-k:i+k]**2)
    e = e/(2*k+1)
    return e
            
        
plt.figure()
plt.plot(data[0,:])
plt.figure(figsize=(20,4))
energy = []
for i in range(0,6,1):
    energy.append(data2energy(data[i,:]))
energy = np.array(energy)

plt.imshow(energy[:,:],aspect='auto')
plt.colorbar()
print('energy.mean = %f ' % energy.mean())

plt.figure()
energy[energy<0.00005]=np.nan
plt.imshow(energy[:,:],aspect='auto')
plt.colorbar()


# In[] 
from scipy import ndimage,signal
import numpy as np
import matplotlib.pyplot as plt
import imageio
from cv2 import cv2


def DiscreteRadonTransform(image, steps):
    channels = len(image[0])
    res = np.zeros((channels, channels), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image, -s*180/steps, reshape=False).astype('float64')
        #print(sum(rotation).shape)
        res[:,s] = sum(rotation)
    return res

plt.figure(figsize=(10,8))
#resampling
a = signal.resample(energy,2000,axis=1)
plt.imshow(a,aspect='auto')
#image = cv2.imread("ct.png", cv2.IMREAD_GRAYSCALE)
image = a
radon = DiscreteRadonTransform(image, len(image[0]))
print(radon.shape)

#绘制原始图像和对应的sinogram图
plt.figure(figsize=(10,8))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray',aspect='auto')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(radon[:,:15], cmap='gray',aspect='auto')
plt.colorbar()
plt.show()


# In[]
# https://blog.csdn.net/God_WZH/article/details/122223501

from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter
SECONDS_PER_DAY = 3600.0 * 24.0

plt.figure(figsize=(15,5))
ax = plt.subplot(111)
trace =  st[0]
x_values = ((trace.times() / SECONDS_PER_DAY) + date2num(trace.stats.starttime.datetime))
ax.plot(x_values,trace.data)
formatter = DateFormatter('%D:%H:%M')
ax.xaxis.set_major_formatter(formatter) 
 
 
 
 
 
 