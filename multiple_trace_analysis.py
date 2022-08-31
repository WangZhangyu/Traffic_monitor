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
