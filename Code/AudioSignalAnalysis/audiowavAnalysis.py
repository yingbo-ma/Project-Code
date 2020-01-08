#! usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20

@author: yingbo.ma@ufl.edu
"""
#%%
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

(sample_rate, sig) = wavfile.read('/home/yingbo/FlecksCode/python/AudioSignalAnalysis/signal.wav')
sample_points = sig.shape[0]
print("sample_rate: %d" % sample_rate)
print("sample_points: %d" % sample_points)
print("time_last = sample_points/sample_rate")

# data_0 = sig[:, 0]
data_0 = sig
time = np.arange(0, sample_points) * (1 / sample_rate)

fig = plt.figure('Figure1').add_subplot(211)
fig.plot(time, data_0)
fig.set_xlabel("time/seconds")
fig.set_ylabel("Amplitude")

k = np.arange(len(data_0))
T = len(data_0)/sample_rate
freq = k/T

DATA_0 = np.fft.fft(data_0)
abs_DATA_0 = abs(DATA_0)

fig = plt.figure('Figure2').add_subplot(212)
fig.plot(freq, abs_DATA_0)
fig.set_xlabel("Frequency/Hz")
fig.set_ylabel("Amplitude")

plt.show()
#%%
